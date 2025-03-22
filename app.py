#!/usr/bin/env python3
"""
PDF to Questions Web Interface

A web-based application for educational content processing that provides:
1. Document Upload: Secure PDF file submission
2. Configuration Options: Customizable processing parameters
3. Result Retrieval: Organized spreadsheet download capability
"""

import os
import time
import uuid
import logging
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
import tempfile
from werkzeug.utils import secure_filename

# Import the question extraction functionality
from pdf_to_questions import convert_pdf_to_images, extract_text_with_gemini, extract_questions_with_gemini
from pdf_to_questions import improve_questions, process_pdf_page

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Create required directories
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables.")
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Track job status
jobs = {}

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    if 'pdf_file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('index'))
    
    file = request.files['pdf_file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    try:
        # Generate a unique ID for this job
        job_id = str(uuid.uuid4())
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
        file.save(file_path)
        
        # Get parameters from form
        start_page = int(request.form.get('start_page', 1))
        max_pages = int(request.form.get('max_pages', 0))
        
        # Initialize job status
        jobs[job_id] = {
            'status': 'processing',
            'progress': 0,
            'file_path': file_path,
            'start_page': start_page,
            'max_pages': max_pages,
            'current_page': 0,
            'total_pages': 0,
            'questions': [],
            'error': None
        }
        
        # Process in background
        import threading
        thread = threading.Thread(target=process_pdf, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        return redirect(url_for('job_status', job_id=job_id))
    
    except Exception as e:
        logging.error(f"Error processing upload: {str(e)}")
        flash(f"Error processing file: {str(e)}")
        return redirect(url_for('index'))

@app.route('/status/<job_id>')
def job_status(job_id):
    """Show job status page"""
    if job_id not in jobs:
        flash('Job not found')
        return redirect(url_for('index'))
    
    return render_template('job_status.html', job=jobs[job_id])

@app.route('/api/status/<job_id>')
def api_status(job_id):
    """Return job status as JSON"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(jobs[job_id])

@app.route('/download/<job_id>')
def download_file(job_id):
    """Download the processed Excel file"""
    if job_id not in jobs or jobs[job_id]['status'] != 'completed':
        flash('File not ready for download')
        return redirect(url_for('index'))
    
    try:
        excel_path = os.path.join(RESULTS_FOLDER, f"{job_id}.xlsx")
        return send_file(
            excel_path,
            as_attachment=True,
            download_name='extracted_questions.xlsx'
        )
    except Exception as e:
        logging.error(f"Error downloading file: {str(e)}")
        flash(f"Error downloading file: {str(e)}")
        return redirect(url_for('index'))

def process_pdf(job_id):
    """Process the PDF file and extract questions"""
    job = jobs[job_id]
    
    try:
        # Convert pages to images
        image_paths = convert_pdf_to_images(
            job['file_path'],
            start_page=job['start_page'],
            max_pages=job['max_pages']
        )
        
        # Process each image
        for i, img_path in enumerate(image_paths):
            # Extract text
            text = extract_text_with_gemini(img_path, GEMINI_API_KEY)
            
            # Extract questions
            questions = extract_questions_with_gemini(text, GEMINI_API_KEY)
            
            # Add page numbers
            for q in questions:
                q['page_number'] = job['start_page'] + i
            
            # Improve questions
            questions = improve_questions(questions, GEMINI_API_KEY)
            
            # Add to results
            job['questions'].extend(questions)
            
            # Update progress
            job['current_page'] = i + 1
            job['progress'] = ((i + 1) / len(image_paths)) * 100
        
        # Save to Excel
        import pandas as pd
        excel_path = os.path.join(RESULTS_FOLDER, f"{job_id}.xlsx")
        df = pd.DataFrame(job['questions'])
        df.to_excel(excel_path, index=False)
        
        # Update job status
        job['status'] = 'completed'
        job['progress'] = 100
        
        # Clean up uploaded file
        try:
            os.remove(job['file_path'])
        except Exception as e:
            logging.warning(f"Failed to remove uploaded file: {str(e)}")
        
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        job['status'] = 'failed'
        job['error'] = str(e)
        
        # Clean up on error
        try:
            os.remove(job['file_path'])
        except:
            pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 