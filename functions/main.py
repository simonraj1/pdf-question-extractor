# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

import os
import tempfile
from firebase_functions import https_fn
from firebase_admin import initialize_app, storage
from flask import Flask, request
import google.generativeai as genai

# Initialize Firebase Admin
app = initialize_app()

# Initialize Firebase Storage bucket
bucket = storage.bucket()

# Initialize Flask app
flask_app = Flask(__name__)

@https_fn.on_request()
def handle_request(req: https_fn.Request) -> https_fn.Response:
    """Handle incoming requests."""
    try:
        # Get the path from the request
        path = req.path

        if path == '/':
            return 'PDF Question Extractor API is running!'
            
        elif path == '/upload' and req.method == 'POST':
            if 'file' not in req.files:
                return 'No file uploaded', 400
                
            file = req.files['file']
            if file.filename == '':
                return 'No file selected', 400
                
            if not file.filename.lower().endswith('.pdf'):
                return 'Invalid file type. Please upload a PDF file.', 400
                
            # Process the file
            try:
                # Save file temporarily
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, file.filename)
                file.save(temp_path)
                
                # Upload to Firebase Storage
                blob = bucket.blob(f'uploads/{file.filename}')
                blob.upload_from_filename(temp_path)
                
                # Clean up
                os.remove(temp_path)
                os.rmdir(temp_dir)
                
                return {'message': 'File uploaded successfully', 'url': blob.public_url}
                
            except Exception as e:
                return f'Error processing file: {str(e)}', 500
                
        else:
            return f'Invalid endpoint: {path}', 404
            
    except Exception as e:
        return f'Server error: {str(e)}', 500

if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))