from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from chatbot_logic import handle_user_query

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    """
    API endpoint to handle user queries.
    Expects a JSON payload with a 'query' field.
    """
    try:
        data = request.get_json()
        user_query = data.get('query')
        
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Handle the user query using chatbot logic
        response = handle_user_query(user_query)
        
        return jsonify({
            'response': response
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
