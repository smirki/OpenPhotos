import os
import requests
import time
import threading
import sqlite3
from dotenv import load_dotenv
import base64
import json


load_dotenv()

OLLAMA_API_URL = 'http://localhost:11434/api/generate'
MODEL_NAME = 'llava'
DATABASE_PATH = 'photo_data.db'
IMAGE_FOLDER = os.getenv('IMAGE_FOLDER')
SERVER_URL = 'http://localhost:5000' 


total_images = 0
tagged_images = 0
running = True

def image_to_base64(image_path):
    print(f"Converting image {image_path} to base64.")
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            print(f"Successfully converted image {image_path} to base64.")
            return encoded_string
    except Exception as e:
        print(f"Error converting image {image_path} to base64: {e}")
        return ""

def generate_tags(image_base64):
    print("Sending request to Ollama API for tag generation.")
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
        print(f"Ollama API response received: {api_response}")
        
        time.sleep(0.3)
        if 'response' in api_response:
            tags = [tag.strip() for tag in api_response['response'].split(',')]
            print(f"Generated tags: {tags}")
            return tags
        else:
            print("No tags found in API response.")
            return []
    except requests.RequestException as e:
        print(f"Error generating tags from image: {e}")
        return []

def notify_server(tagged_images, total_images):
    print(f"Notifying server about progress: {tagged_images}/{total_images} images tagged.")
    try:
        response = requests.post(f'{SERVER_URL}/update_tagging_progress', json={
            'tagged_images': tagged_images,
            'total_images': total_images
        })
        if response.status_code == 200:
            print("Server notified successfully.")
        else:
            print(f"Failed to notify server. Status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error notifying server: {e}")

def process_images():
    global total_images, tagged_images, running
    
    print("Connecting to the SQLite database.")
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    print("Fetching untagged photos from the database.")
    cursor.execute("SELECT id, path FROM photos WHERE tags IS '[]'")
    untagged_photos = cursor.fetchall()
    total_images = len(untagged_photos)
    print(f"Total untagged images to process: {total_images}")
    
    for photo_id, file_path in untagged_photos:
        if not running:
            print("Tagging process stopped by user.")
            break
        
        try:
            print(f"Processing image: {file_path} (ID: {photo_id})")
            image_base64 = image_to_base64(file_path)
            if image_base64:
                tags = generate_tags(image_base64)
                tags_json = json.dumps(tags)

                print(f"Updating database with tags for image ID: {photo_id}")
                cursor.execute("UPDATE photos SET tags = ? WHERE id = ?", (tags_json, photo_id))
                conn.commit()
                
                tagged_images += 1
                print(f"Tagged {tagged_images}/{total_images} images.")
                
                notify_server(tagged_images, total_images)
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")

    print("Closing the database connection.")
    conn.close()

def start_tagging():
    global running
    print("Starting the tagging process.")
    running = True
    tagging_thread = threading.Thread(target=process_images)
    tagging_thread.start()

def stop_tagging():
    global running
    print("Stopping the tagging process.")
    running = False
