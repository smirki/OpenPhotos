import os
import sqlite3
from dotenv import load_dotenv
import threading
import json
import numpy as np
from flask import Flask, render_template, request, url_for, send_from_directory, jsonify
import tagging_worker  
import face_detection_worker  
import requests
import time
import random

load_dotenv()

app = Flask(__name__)
DATABASE_PATH = 'photo_data.db'  
IMAGE_FOLDER = os.getenv('IMAGE_FOLDER')
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:latest"

face_detection_status = {
    "total_images": 0,
    "processed_images": 0
}

tagging_status = {
    "total_images": 0,
    "tagged_images": 0
}

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
    print("Sending request to Ollama API for tag generation.")
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": f"turn '{search_phrase}' into tags as if you were tagging and classifying an image. Only reply with the tags, comma separated, nothing else.",
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        api_response = response.json()
        print(f"Ollama API response received: {api_response}")
        
        time.sleep(0.3)
        if 'response' in api_response:
            tags = [tag.strip().lower() for tag in api_response['response'].split(',')]
            print(f"Generated tags: {tags}")
            return tags
        else:
            print("No tags found in API response.")
            return []
    except requests.RequestException as e:
        print(f"Error generating tags from image: {e}")
        return []

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
        print(f"AI Tags: {ai_tags}") 

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
        print(f"Error during face search: {e}")
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
        print(f"Using precomputed embedding for {photo['path']} with face coordinates: {facial_area}")
    else:
        print(f"No precomputed embeddings found for {photo['path']}")

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
            print(f"Error parsing GPS data for {photo_path}: {e}")
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

if __name__ == '__main__':
    photos = load_photos()
    face_count = count_images_with_faces(photos)
    print(f"Number of images with detected faces: {face_count}")
    
    app.run(debug=True)
