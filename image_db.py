import os
import sqlite3
import datetime
import base64
import requests
import exifread
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import time
import sys
import subprocess
import ffmpeg
import pillow_heif
from deepface import DeepFace
import tqdm
import json

load_dotenv()


OLLAMA_API_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'llava'
DATABASE_PATH = 'photo_data.db'  
IMAGE_FOLDER = os.getenv('IMAGE_FOLDER')
THUMBNAILS_FOLDER = os.getenv('THUMBNAILS_FOLDER')


ENABLE_AI_TAGS = '--tags' in sys.argv
ENABLE_FACE_RECOGNITION = '--face' in sys.argv

def setup_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS photos (
                        id INTEGER PRIMARY KEY,
                        path TEXT UNIQUE,
                        thumbnail TEXT,
                        date TEXT,
                        metadata TEXT,
                        tags TEXT,
                        embedding BLOB,
                        face_coords TEXT
                    )''')
    conn.commit()
    return conn

def convert_exif_data(exif_data):
    serializable_data = {}
    for tag, value in exif_data.items():
        if isinstance(value, (list, tuple)):
            serializable_data[tag] = [str(v) for v in value]
        elif isinstance(value, bytes):
            serializable_data[tag] = value.decode('utf-8', 'ignore')
        elif hasattr(value, 'isoformat'):
            serializable_data[tag] = value.isoformat()
        else:
            serializable_data[tag] = str(value)
    return serializable_data

def extract_exif_data(file_path):
    try:
        if file_path.lower().endswith('.heic'):
            heif_file = pillow_heif.read_heif(file_path)
            exif_data = heif_file.info.get('exif', {})
        else:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=True)
            exif_data = convert_exif_data(tags)
        return exif_data
    except Exception as e:
        print(f"Error extracting EXIF data from {file_path}: {e}")
        return {}

def get_image_date(exif_data, file_path):
    date = exif_data.get('EXIF DateTimeOriginal') or exif_data.get('Image DateTime')
    if date:
        try:
            date_str = str(date)
            date_time = datetime.datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
            return date_time.isoformat()
        except Exception as e:
            print(f"Error parsing date for {file_path}: {e}")
            return datetime.datetime.now().isoformat()
    else:
        mod_time = os.path.getmtime(file_path)
        return datetime.datetime.fromtimestamp(mod_time).isoformat()

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error converting image {image_path} to base64: {e}")
        return ""

def generate_tags(image_base64):
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": "You are an image tagging bot. Provide a comprehensive list of tags for this image, separated by commas. Do not include any other text in your response.",
            "images": [image_base64],
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        api_response = response.json()
        print(f"API Response: {api_response}")
        
        time.sleep(0.3)
        if 'response' in api_response:
            tags = [tag.strip() for tag in api_response['response'].split(',')]
            return tags
        else:
            return []
    except requests.RequestException as e:
        print(f"Error generating tags from image: {e}")
        return []

def create_thumbnail(image_path, thumbnail_path):
    try:
        img = Image.open(image_path)
        img.thumbnail((150, 150))
        thumbnail_path = thumbnail_path.replace('\\', '/')
        img.save(thumbnail_path)
    except Exception as e:
        print(f"Error creating thumbnail for {image_path}: {e}")

def convert_heic_to_jpg(image_path):
    try:
        output_path = image_path.replace('.heic', '.jpg').replace('.HEIC', '.jpg')
        heif_file = pillow_heif.read_heif(image_path)
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data, 
            "raw", 
            heif_file.mode, 
            heif_file.stride,
        )
        image.save(output_path, format="JPEG")
        return output_path
    except Exception as e:
        print(f"Error converting HEIC image {image_path} to JPG: {e}")
        return image_path

def extract_frame_from_mov(image_path):
    try:
        output_path = image_path.replace('.mov', '.jpg').replace('.MOV', '.jpg')
        process = (
            ffmpeg
            .input(image_path, ss='00:00:01')
            .output(output_path, vframes=1)
            .overwrite_output()
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        stdout, stderr = process.communicate(timeout=10) 
        if process.returncode != 0:
            raise Exception(f"ffmpeg error: {stderr.decode('utf-8')}")
        return output_path
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"Timeout while processing {image_path}")
        return None
    except Exception as e:
        print(f"Error extracting frame from MOV file {image_path}: {e}")
        return None

def process_live_images(image_path):
    if image_path.lower().endswith('.heic'):
        return convert_heic_to_jpg(image_path)
    elif image_path.lower().endswith('.mov'):

        return extract_frame_from_mov(image_path)
    return None

def precompute_face_embeddings(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM photos WHERE embedding IS NULL")
    photos = cursor.fetchall()

    for idx, photo in enumerate(tqdm.tqdm(photos, desc="Processing images", unit="image")):
        img_path = photo[1] 
        try:
            if img_path.lower().endswith(('.arw', '.raw')):
                continue

            faces = DeepFace.extract_faces(img_path=img_path, enforce_detection=False)
            
            if faces: 
                embeddings = DeepFace.represent(img_path=img_path, model_name="VGG-Face", enforce_detection=False)
                
                if embeddings:
                    with Image.open(img_path) as img:
                        width, height = img.size

                    face_coords = faces[0]['facial_area']
                    face_coords['x'] = max(0, min(face_coords['x'], width))
                    face_coords['y'] = max(0, min(face_coords['y'], height))
                    face_coords['w'] = max(0, min(face_coords['w'], width - face_coords['x']))
                    face_coords['h'] = max(0, min(face_coords['h'], height - face_coords['y']))

                    cursor.execute('''UPDATE photos SET embedding=?, face_coords=? WHERE id=?''',
                                   (json.dumps(embeddings[0]['embedding']),
                                    json.dumps(face_coords),
                                    photo[0]))
                    conn.commit()
                else:
                    print(f"No embeddings found for {img_path}")
            else:
                print(f"No faces detected in {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def index_photos(folder_path, conn):
    print(f"Starting indexing of folder: {folder_path}")
    total_files = sum(1 for root, _, files in os.walk(folder_path) 
                      for file in files if file.lower().endswith(('.jpg', '.jpeg', '.heic', '.mov')))
    
    if total_files == 0:
        print("No images found to process.")
        return

    print(f"Found {total_files} images to process.")
    processed_files = 0
    start_time = time.time()

    cursor = conn.cursor()
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.heic', '.mov')):
                file_path = os.path.join(root, file)
                try:
                    if file.lower().endswith(('.mov', '.heic')):
                        file_path = process_live_images(file_path)
                        if not file_path:
                            continue
                    
                    exif_data = extract_exif_data(file_path)
                    
                    image_date = get_image_date(exif_data, file_path)
                    
                    tags = []
                    if ENABLE_AI_TAGS:
                        image_base64 = image_to_base64(file_path)
                        if image_base64:
                            tags = generate_tags(image_base64)
                    
                    thumbnail_path = os.path.join(THUMBNAILS_FOLDER, os.path.basename(file_path))
                    create_thumbnail(file_path, thumbnail_path)
                    
                    cursor.execute('''INSERT OR IGNORE INTO photos (path, thumbnail, date, metadata, tags) 
                                      VALUES (?, ?, ?, ?, ?)''',
                                   (file_path.replace('\\', '/'),
                                    thumbnail_path.replace('\\', '/'),
                                    image_date,
                                    json.dumps(exif_data),
                                    json.dumps(tags)))
                    conn.commit()
                    
                    processed_files += 1
                    elapsed_time = time.time() - start_time
                    avg_time_per_file = elapsed_time / processed_files
                    time_left = (total_files - processed_files) * avg_time_per_file
                    print(f"Processed {processed_files}/{total_files} files. "
                          f"Estimated time left: {time_left/60:.2f} minutes.")
                    
                except Exception as e:
                    print(f"Error processing image {file_path}: {e}")

    if ENABLE_FACE_RECOGNITION:
        print("Precomputing facial embeddings...")
        precompute_face_embeddings(conn)

if __name__ == "__main__":
    conn = setup_database()
    index_photos(IMAGE_FOLDER, conn)
    conn.close()
