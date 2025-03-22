PDF Question Extractor

A web application that extracts and improves multiple-choice questions from PDF documents using advanced AI technology.

Features

- Document Upload: Secure PDF file submission
- Text Recognition: Advanced AI-powered text extraction
- Question Processing: Identifies and structures educational assessment items
- Quality Enhancement: Refines and standardizes question formatting
- Excel Export: Generates organized spreadsheet output
- Real-time Progress: Shows processing status and progress

Requirements

- Python 3.9+
- Required Python packages (installed via setup process):
  - Flask
  - Gunicorn
  - Google Generative AI
  - Pandas
  - PDF2Image
  - Pillow
  - And more (see requirements.txt)

Setup

1. Clone the repository:
```bash
git clone https://github.com/simonraj1/pdf-question-extractor.git
cd pdf-question-extractor
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
GEMINI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
python app.py
```

The application will be available at http://localhost:10000

Deployment

This application is configured for deployment on Render.com. See render.yaml for configuration details.

Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

License

This project is licensed under the MIT License. 