import os
import sqlite3
import threading
import json
import time
from deepface import DeepFace
from PIL import Image
import requests

DATABASE_PATH = 'photo_data.db'
IMAGE_FOLDER = os.getenv('IMAGE_FOLDER')
SERVER_URL = 'http://localhost:5000'

total_images = 0
processed_images = 0
running = True

def detect_faces(image_path):
    try:
        print(f"Detecting faces in {image_path}.")
        faces = DeepFace.extract_faces(img_path=image_path, enforce_detection=False)
        if faces:
            embeddings = DeepFace.represent(img_path=image_path, model_name="VGG-Face", enforce_detection=False)
            if embeddings:
                with Image.open(image_path) as img:
                    width, height = img.size

                face_coords = faces[0]['facial_area']
                face_coords['x'] = max(0, min(face_coords['x'], width))
                face_coords['y'] = max(0, min(face_coords['y'], height))
                face_coords['w'] = max(0, min(face_coords['w'], width - face_coords['x']))
                face_coords['h'] = max(0, min(face_coords['h'], height - face_coords['y']))

                return embeddings[0]['embedding'], face_coords
        print(f"No faces detected in {image_path}.")
        return None, None
    except Exception as e:
        print(f"Error detecting faces in {image_path}: {e}")
        return None, None

def notify_server(processed_images, total_images):
    print(f"Notifying server about progress: {processed_images}/{total_images} images processed.")
    try:
        response = requests.post(f'{SERVER_URL}/update_face_detection_progress', json={
            'processed_images': processed_images,
            'total_images': total_images
        })
        if response.status_code == 200:
            print("Server notified successfully.")
        else:
            print(f"Failed to notify server. Status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error notifying server: {e}")

def process_images():
    global total_images, processed_images, running
    
    print("Connecting to the SQLite database.")
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    print("Fetching images from the database that haven't been processed for face detection.")
    cursor.execute("SELECT id, path FROM photos WHERE embedding IS NULL")
    unprocessed_photos = cursor.fetchall()
    total_images = len(unprocessed_photos)
    print(f"Total images to process: {total_images}")
    
    for photo_id, file_path in unprocessed_photos:
        if not running:
            print("Face detection process stopped by user.")
            break
        
        try:
            print(f"Processing image: {file_path} (ID: {photo_id})")
            embeddings, face_coords = detect_faces(file_path)
            if embeddings:
                print(f"Updating database with face data for image ID: {photo_id}")
                cursor.execute("UPDATE photos SET embedding = ?, face_coords = ? WHERE id = ?",
                               (json.dumps(embeddings), json.dumps(face_coords), photo_id))
                conn.commit()

            processed_images += 1
            print(f"Processed {processed_images}/{total_images} images.")
            
            notify_server(processed_images, total_images)
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")

    print("Closing the database connection.")
    conn.close()

def start_face_detection():
    global running
    print("Starting the face detection process.")
    running = True
    detection_thread = threading.Thread(target=process_images)
    detection_thread.start()

def stop_face_detection():
    global running
    print("Stopping the face detection process.")
    running = False
