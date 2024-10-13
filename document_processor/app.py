from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv  # Import dotenv to load .env
from process_documents import process_and_store_document

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

@app.route('/process_document', methods=['POST'])
def process_document():
    """
    Endpoint to process a document, generate embeddings, and store them in the vector database.
    Expects a JSON payload with the file path of the document to process.
    """
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Invalid file path'}), 400
        
        # Process the document and get the number of document chunks processed
        num_chunks = process_and_store_document(file_path)
        
        return jsonify({
            'message': f'Document processed successfully. Number of chunks: {num_chunks}'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
