from flask import Flask, request, jsonify
import openai
import numpy as np
import faiss
import os
# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the FAISS index and paragraphs
index = faiss.read_index('faiss_index_ada002_large.index')
processed_paragraphs = np.load('paragraphs_ada002_large.npy', allow_pickle=True)

# Function to generate embeddings for a query
def get_gpt4_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return np.array(response['data'][0]['embedding'])

# Function to search the FAISS index
def search_faiss_index(query, k=5):
    query_embedding = get_gpt4_embedding(query).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)  # k = number of results to return
    results = [processed_paragraphs[idx] for idx in indices[0]]
    return results

# Function to generate a final answer based on retrieved paragraphs
def generate_answer(query, retrieved_paragraphs):
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
    return answer

# Define API endpoint for querying the model
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Search the FAISS index and generate an answer
    retrieved_paragraphs = search_faiss_index(query)
    answer = generate_answer(query, retrieved_paragraphs)
    
    return jsonify({"answer": answer})

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
