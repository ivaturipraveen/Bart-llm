import logging
from flask import Flask, request, jsonify
import openai
import numpy as np
import faiss
import os

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Set log level to DEBUG
logger = logging.getLogger(__name__)

# Load the FAISS index and paragraphs
try:
    logger.info("Loading FAISS index and processed paragraphs...")
    index = faiss.read_index('faiss_index_ada002_large.index')
    processed_paragraphs = np.load('paragraphs_ada002_large.npy', allow_pickle=True)
    logger.info("FAISS index and paragraphs loaded successfully.")
except Exception as e:
    logger.error(f"Error loading FAISS index or paragraphs: {e}")
    raise

# Function to generate embeddings for a query
def get_gpt4_embedding(text):
    try:
        logger.info("Generating embedding for query...")
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        embedding = np.array(response['data'][0]['embedding'])
        logger.info("Embedding generated successfully.")
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

# Function to search the FAISS index
def search_faiss_index(query, k=5):
    try:
        logger.info(f"Searching FAISS index for query: {query}")
        query_embedding = get_gpt4_embedding(query).astype('float32').reshape(1, -1)
        distances, indices = index.search(query_embedding, k)  # k = number of results to return
        results = [processed_paragraphs[idx] for idx in indices[0]]
        logger.info(f"Found {len(results)} results for query.")
        return results
    except Exception as e:
        logger.error(f"Error searching FAISS index: {e}")
        raise

# Function to generate a final answer based on retrieved paragraphs
def generate_answer(query, retrieved_paragraphs):
    try:
        logger.info("Generating final answer from retrieved paragraphs...")
        context = "\n\n".join(retrieved_paragraphs)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer the question based on the following information:\n\n{context}\n\nQuestion: {query}"}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=messages,
            max_tokens=150,
            temperature=0.5
        )
        answer = response.choices[0].message['content'].strip()
        logger.info("Answer generated successfully.")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise

# Define API endpoint for querying the model
@app.route('/ask', methods=['POST'])
def ask():
    try:
        logger.info("Received request to '/ask' endpoint.")
        data = request.json
        query = data.get("query", "")
        
        if not query:
            logger.warning("No query provided in the request.")
            return jsonify({"error": "Query is required"}), 400
        
        # Search the FAISS index and generate an answer
        retrieved_paragraphs = search_faiss_index(query)
        answer = generate_answer(query, retrieved_paragraphs)
        
        logger.info("Returning the generated answer.")
        return jsonify({"answer": answer})
    
    except Exception as e:
        logger.error(f"Error processing the request: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# Run the Flask app
if __name__ == '__main__':
    logger.info("Starting Flask app on host 0.0.0.0, port 5000...")
    app.run(host="0.0.0.0", port=5000)
