import os
import sqlite3
import datetime
import json
import logging
import threading
import time
import random
from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory, jsonify
from PIL import Image, ExifTags

import tagging_worker  
import face_detection_worker  
import requests
import numpy as np

# -----------------------------------------------------------------------------
# Setup logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Load environment variables from .env
# -----------------------------------------------------------------------------
load_dotenv(verbose=True, override=True)
IMAGE_FOLDER = os.getenv('IMAGE_FOLDER')
THUMBNAILS_FOLDER = os.getenv('THUMBNAILS_FOLDER')

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
app = Flask(__name__)
DATABASE_PATH = 'photo_data.db'
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi4:latest"

face_detection_status = {
    "total_images": 0,
    "processed_images": 0
}

tagging_status = {
    "total_images": 0,
    "tagged_images": 0
}

# -----------------------------------------------------------------------------
# Database Initialization
# -----------------------------------------------------------------------------
def init_db():
    """
    Initialize the database. Creates the photos table if it does not exist.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            thumbnail TEXT,
            date TEXT,
            metadata TEXT,
            tags TEXT,
            embedding BLOB,
            face_coords TEXT
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized.")

# -----------------------------------------------------------------------------
# Helper – Create Thumbnail for an Image
# -----------------------------------------------------------------------------
def create_thumbnail(image_path):
    """
    Generate a thumbnail for the given image and save it in THUMBNAILS_FOLDER.
    Returns the thumbnail file path (or an empty string on error).
    """
    try:
        if not THUMBNAILS_FOLDER:
            logger.error("THUMBNAILS_FOLDER is not set in the .env file.")
            return ""
        if not os.path.exists(THUMBNAILS_FOLDER):
            os.makedirs(THUMBNAILS_FOLDER)
            logger.info(f"Created thumbnails folder at: {THUMBNAILS_FOLDER}")
        img = Image.open(image_path)
        img.thumbnail((150, 150))
        thumbnail_filename = os.path.basename(image_path)
        thumbnail_path = os.path.join(THUMBNAILS_FOLDER, thumbnail_filename)
        img.save(thumbnail_path)
        logger.info(f"Thumbnail created at: {thumbnail_path}")
        return thumbnail_path.replace('\\', '/')
    except Exception as e:
        logger.error(f"Error creating thumbnail for {image_path}: {e}")
        return ""

# -----------------------------------------------------------------------------
# Ingest Photos from a Specified Folder
# -----------------------------------------------------------------------------
def ingest_photos_in_folder(folder_path):
    """
    Walk through the given folder (and its subfolders) and add any new image file to the database.
    For each image, extract the file’s modification date, attempt to read EXIF data, generate a thumbnail,
    and then insert a record into the database.
    """
    folder_logger = logging.getLogger("ingest_photos_in_folder")
    if not folder_path or not os.path.exists(folder_path):
        folder_logger.error(f"Folder path '{folder_path}' does not exist.")
        return

    folder_logger.info(f"Starting ingestion of photos from folder: {folder_path}")
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    count_indexed = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.mov')):
                file_path = os.path.join(root, file).replace('\\', '/')
                # Skip if already indexed
                cursor.execute("SELECT id FROM photos WHERE path=?", (file_path,))
                if cursor.fetchone():
                    folder_logger.debug(f"Already indexed: {file_path}")
                    continue

                # Use file modification time as the date
                try:
                    mod_time = os.path.getmtime(file_path)
                    date = datetime.datetime.fromtimestamp(mod_time).isoformat()
                except Exception as e:
                    folder_logger.error(f"Error getting modification time for {file_path}: {e}")
                    date = datetime.datetime.now().isoformat()

                # Attempt to extract EXIF data
                metadata = {}
                try:
                    img = Image.open(file_path)
                    exif = img._getexif()
                    if exif:
                        metadata = {ExifTags.TAGS.get(k, k): v for k, v in exif.items() if k in ExifTags.TAGS}
                except Exception as e:
                    folder_logger.error(f"Error extracting EXIF from {file_path}: {e}")

                # Generate a thumbnail for the image
                thumbnail = create_thumbnail(file_path)

                # Insert record into the database
                try:
                    cursor.execute(
                        "INSERT INTO photos (path, thumbnail, date, metadata, tags, embedding, face_coords) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (file_path, thumbnail, date, json.dumps(metadata), json.dumps([]), None, json.dumps({}))
                    )
                    conn.commit()
                    count_indexed += 1
                    folder_logger.info(f"Ingested photo: {file_path}")
                except Exception as e:
                    folder_logger.error(f"Error inserting {file_path} into database: {e}")
    conn.close()
    folder_logger.info(f"Ingestion complete. {count_indexed} new photos ingested from: {folder_path}")

# -----------------------------------------------------------------------------
# Generate Missing Thumbnails
# -----------------------------------------------------------------------------
def generate_missing_thumbnails():
    """
    Find photos in the database that do not have a thumbnail and generate one.
    This function is intended to run in a background thread.
    """
    thumb_logger = logging.getLogger("generate_missing_thumbnails")
    thumb_logger.info("Starting missing thumbnail generation.")
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    # Look for records with a NULL or empty thumbnail
    cursor.execute("SELECT id, path FROM photos WHERE thumbnail IS NULL OR thumbnail = ''")
    photos = cursor.fetchall()
    count = 0

    for photo_id, photo_path in photos:
        thumbnail = create_thumbnail(photo_path)
        if thumbnail:
            try:
                cursor.execute("UPDATE photos SET thumbnail=? WHERE id=?", (thumbnail, photo_id))
                conn.commit()
                count += 1
                thumb_logger.info(f"Generated thumbnail for {photo_path}")
            except Exception as e:
                thumb_logger.error(f"Error updating thumbnail for {photo_path}: {e}")
    conn.close()
    thumb_logger.info(f"Thumbnail generation complete. {count} thumbnails generated.")

# -----------------------------------------------------------------------------
# Load Photos from the Database (existing code)
# -----------------------------------------------------------------------------
def load_photos():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM photos")
    photos = cursor.fetchall()
    conn.close()

    photo_list = []
    for photo in photos:
        metadata = json.loads(photo[4])
        tags = json.loads(photo[5]) if photo[5] else None
        embedding = json.loads(photo[6]) if photo[6] else None
        face_coords = json.loads(photo[7]) if photo[7] else None

        pills = []
        if metadata:
            pills.append('EXIF')
        if 'GPS GPSLatitude' in metadata and 'GPS GPSLongitude' in metadata:
            pills.append('GEO')
        file_extension = os.path.splitext(photo[1])[1].upper().replace('.', '')
        if file_extension in ['RAW', 'NEF', 'CR2']:
            pills.append('RAW')
        pills.append(file_extension)
        if embedding:
            pills.append('FACE')
        if tags:
            pills.append('TAGS')

        photo_data = {
            'id': photo[0],
            'path': photo[1],
            'thumbnail': photo[2],
            'date': photo[3],
            'metadata': metadata,
            'tags': tags,
            'embedding': embedding,
            'face_coords': face_coords,
            'pills': pills
        }
        photo_list.append(photo_data)
    return photo_list

def count_images_with_faces(photos):
    return sum(1 for photo in photos if photo['embedding'])

def generate_tags_with_ollama(search_phrase):
    logger.info("Sending request to Ollama API for tag generation.")
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": f"turn '{search_phrase}' into tags as if you were tagging and classifying an image. Only reply with the tags, comma separated, nothing else.",
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        api_response = response.json()
        logger.info(f"Ollama API response received: {api_response}")
        
        time.sleep(0.3)
        if 'response' in api_response:
            tags = [tag.strip().lower() for tag in api_response['response'].split(',')]
            logger.info(f"Generated tags: {tags}")
            return tags
        else:
            logger.info("No tags found in API response.")
            return []
    except requests.RequestException as e:
        logger.error(f"Error generating tags from image: {e}")
        return []

# -----------------------------------------------------------------------------
# Flask Routes
# -----------------------------------------------------------------------------
@app.route('/start_face_detection', methods=['POST'])
def start_face_detection():
    face_detection_worker.start_face_detection()
    return jsonify({"status": "Face detection started"}), 200

@app.route('/stop_face_detection', methods=['POST'])
def stop_face_detection():
    face_detection_worker.stop_face_detection()
    return jsonify({"status": "Face detection stopped"}), 200

@app.route('/update_face_detection_progress', methods=['POST'])
def update_face_detection_progress():
    global face_detection_status
    data = request.json
    face_detection_status["processed_images"] = data.get("processed_images", 0)
    face_detection_status["total_images"] = data.get("total_images", 0)
    return jsonify({"status": "Progress updated"}), 200

@app.route('/face_detection_status')
def face_detection_status_route():
    return jsonify(face_detection_status)

@app.route('/start_tagging', methods=['POST'])
def start_tagging():
    tagging_worker.start_tagging()
    return jsonify({"status": "Tagging started"}), 200

@app.route('/stop_tagging', methods=['POST'])
def stop_tagging():
    tagging_worker.stop_tagging()
    return jsonify({"status": "Tagging stopped"}), 200

@app.route('/update_exif/<int:photo_id>', methods=['POST'])
def update_exif(photo_id):
    new_exif = request.form.to_dict()
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT metadata FROM photos WHERE id=?", (photo_id,))
    current_metadata = json.loads(cursor.fetchone()[0])
    current_metadata.update(new_exif)

    cursor.execute("UPDATE photos SET metadata=? WHERE id=?", (json.dumps(current_metadata), photo_id))
    conn.commit()
    conn.close()

    return jsonify({"status": "success"}), 200

@app.route('/update_tags/<int:photo_id>', methods=['POST'])
def update_tags(photo_id):
    new_tags = request.form.getlist('tags')
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("UPDATE photos SET tags=? WHERE id=?", (json.dumps(new_tags), photo_id))
    conn.commit()
    conn.close()

    return jsonify({"status": "success"}), 200

@app.route('/update_tagging_progress', methods=['POST'])
def update_tagging_progress():
    global tagging_status
    data = request.json
    tagging_status["tagged_images"] = data.get("tagged_images", 0)
    tagging_status["total_images"] = data.get("total_images", 0)
    return jsonify({"status": "Progress updated"}), 200

@app.route('/tagging_status')
def tagging_status_route():
    return jsonify(tagging_status)

@app.route('/')
def index():
    search_query = request.args.get('search', '').lower()
    photos = load_photos()

    normal_search_results = []
    ai_search_results = []

    if search_query:
        normal_search_results = [
            photo for photo in photos if search_query in ' '.join(photo.get('tags', []) + list(photo.get('metadata', {}).values())).lower()
        ]
        ai_tags = generate_tags_with_ollama(search_query)
        logger.info(f"AI Tags: {ai_tags}")

        if ai_tags:
            ai_search_results = [
                photo for photo in photos if len(set(ai_tags) & set(photo.get('tags', []))) > 1
            ]
    else:
        normal_search_results = photos

    grouped_normal_search = {}
    for photo in normal_search_results:
        date = photo['date']
        month_year = date[:7]
        if month_year not in grouped_normal_search:
            grouped_normal_search[month_year] = []
        grouped_normal_search[month_year].append(photo)

    grouped_ai_search = {}
    for photo in ai_search_results:
        date = photo['date']
        month_year = date[:7]
        if month_year not in grouped_ai_search:
            grouped_ai_search[month_year] = []
        grouped_ai_search[month_year].append(photo)

    return render_template(
        'index.html',
        grouped_photos=grouped_normal_search,
        ai_grouped_photos=grouped_ai_search,
        tagging_status=tagging_status,
        face_detection_status=face_detection_status,
        random=random
    )

@app.route('/search_face', methods=['POST'])
def search_face():
    try:
        photos = load_photos()
        face_embedding = np.array(request.json['embedding'])

        similar_photos = []
        for photo in photos:
            if photo['embedding']:
                photo_embedding = np.array(photo['embedding'])
                distance = np.linalg.norm(face_embedding - photo_embedding)
                if distance < 1:
                    similar_photos.append(photo)

        return jsonify(similar_photos)
    except Exception as e:
        logger.error(f"Error during face search: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/face_search_results', methods=['POST'])
def face_search_results():
    data = request.json
    original_image = data.get('original_image')
    similar_photos = data.get('similar_photos', [])
    return render_template('face_search_results.html', original_image=original_image, similar_photos=similar_photos)

@app.route('/photo/<path:photo_path>')
def view_photo(photo_path):
    photos = load_photos()
    photo = next((p for p in photos if p['path'] == photo_path), None)

    if not photo:
        return "Photo not found", 404

    photo_index = photos.index(photo)
    next_photo = photos[photo_index + 1] if photo_index + 1 < len(photos) else None
    prev_photo = photos[photo_index - 1] if photo_index - 1 >= 0 else None

    detected_faces = []
    if photo['embedding']:
        facial_area = photo.get('face_coords', {'x': 0, 'y': 0, 'w': 100, 'h': 100})
        detected_faces.append({
            'embedding': photo['embedding'],
            'facial_area': facial_area
        })
        logger.info(f"Using precomputed embedding for {photo['path']} with face coordinates: {facial_area}")
    else:
        logger.info(f"No precomputed embeddings found for {photo['path']}")

    gps_lat = photo['metadata'].get('GPS GPSLatitude', None)
    gps_lon = photo['metadata'].get('GPS GPSLongitude', None)

    def parse_gps_coordinate(coord):
        if isinstance(coord, str):
            coord = eval(coord)
        return [float(value) if isinstance(value, (int, float)) else eval(value) for value in coord]

    if gps_lat and gps_lon:
        try:
            gps_lat = parse_gps_coordinate(gps_lat)
            gps_lon = parse_gps_coordinate(gps_lon)
            latitude = gps_lat[0] + gps_lat[1] / 60 + gps_lat[2] / 3600
            longitude = -(gps_lon[0] + gps_lon[1] / 60 + gps_lon[2] / 3600)
        except Exception as e:
            logger.error(f"Error parsing GPS data for {photo_path}: {e}")
            latitude = longitude = None
    else:
        latitude = longitude = None

    return render_template(
        'photo_view.html',
        photo=photo,
        next_photo=next_photo,
        prev_photo=prev_photo,
        detected_faces=detected_faces,
        latitude=latitude,
        longitude=longitude
    )

@app.route('/images/<path:filename>')
def image(filename):
    normalized_path = os.path.normpath(filename)
    directory = os.path.dirname(normalized_path)
    file_name = os.path.basename(normalized_path)
    return send_from_directory(directory, file_name)

# -----------------------------------------------------------------------------
# New Routes for Thumbnail Generation and Photo Ingestion
# -----------------------------------------------------------------------------
@app.route('/generate_thumbnails', methods=['POST'])
def route_generate_thumbnails():
    """Starts a background thread to generate missing thumbnails."""
    threading.Thread(target=generate_missing_thumbnails).start()
    return jsonify({"status": "Thumbnail generation started."}), 200

@app.route('/ingest', methods=['POST'])
def route_ingest():
    """Starts a background thread to ingest photos from the main IMAGE_FOLDER."""
    threading.Thread(target=ingest_photos_in_folder, args=(IMAGE_FOLDER,)).start()
    return jsonify({"status": "Photo ingestion from main folder started."}), 200

@app.route('/ingest_directory', methods=['POST'])
def route_ingest_directory():
    """
    Ingest photos from a directory specified in the POST JSON payload.
    The payload must include a 'directory' key.
    """
    data = request.get_json()
    directory = data.get('directory')
    if not directory or not os.path.exists(directory):
        return jsonify({"error": "Invalid directory"}), 400
    threading.Thread(target=ingest_photos_in_folder, args=(directory,)).start()
    return jsonify({"status": f"Photo ingestion from directory {directory} started."}), 200

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    init_db()
    # Ingest the main folder asynchronously
    threading.Thread(target=ingest_photos_in_folder, args=(IMAGE_FOLDER,)).start()
    # Generate missing thumbnails asynchronously (won't block server startup)
    threading.Thread(target=generate_missing_thumbnails).start()
    app.run(debug=True)
