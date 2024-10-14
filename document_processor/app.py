from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv  # Import dotenv to load .env
from process_documents import process_and_store_document
import logging
import sys

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure logging to log to stdout
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)

@app.route('/process_document', methods=['POST'])
def process_document():
    """
    Endpoint to process a document, generate embeddings, and store them in the vector database.
    Expects a JSON payload with the file path of the document to process.
    """
    app.logger.info("process_document route accessed.")
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path or not os.path.exists(file_path):
            app.logger.error(f"file does not exist: {file_path}")
            return jsonify({'error': 'Invalid file path'}), 400
        app.logger.info("start to process the document.")
        # Process the document and get the number of document chunks processed
        num_chunks = process_and_store_document(file_path)
        
        return jsonify({
            'message': f'Document processed successfully. Number of chunks: {num_chunks}'
        }), 200
    except Exception as e:
        app.logger.error(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
