import sys

import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from huggingface_hub import login

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=huggingface_token)

# Step 1: Load the Llama model for text generation
llama_model_path = 'meta-llama/Llama-3.2-1B'  # Replace with your model path
tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
model = AutoModelForCausalLM.from_pretrained(llama_model_path)

# Step 2: Load and embed AI researcher profiles
# Example profiles (replace with your actual data loading mechanism)
profiles = [
    {
        'name': 'Dr. Alice Smith',
        'expertise': 'Machine Learning, Computer Vision',
        'email': 'alice.smith@example.com',
        'description': 'Expert in deep learning and image analysis.',
    },
    {
        'name': 'Dr. Bob Johnson',
        'expertise': 'Natural Language Processing, Text Mining',
        'email': 'bob.johnson@example.com',
        'description': 'Specializes in language models and text data analysis.',
    },
    # Add more profiles as needed
]

# Initialize the embedding model (runs locally)
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Can be replaced with any local model
embedding_model = SentenceTransformer(embedding_model_name)

# Create embeddings for the profiles
profile_texts = [
    f"{p['name']}. Expertise: {p['expertise']}. {p['description']}" for p in profiles
]
profile_embeddings = embedding_model.encode(profile_texts, convert_to_numpy=True)

# Step 3: Build a FAISS index
embedding_dimension = profile_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)
index.add(profile_embeddings)

# Function to find relevant profiles
def find_relevant_profiles(query, top_k=1):
    # Step 4: Generate embedding for the query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    # Step 5: Search in FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve the corresponding profiles
    relevant_profiles = [profiles[idx] for idx in indices[0]]
    return relevant_profiles

# Function to generate a response
def generate_response(query, relevant_profiles):
    # Build a prompt that includes the profiles
    profiles_text = '\n'.join([
        f"Name: {p['name']}\nUnit: {['research_unit']}\Bio: {p['bio']}\Keywords: {p['keywords']}\n"
        for p in relevant_profiles
    ])
    prompt = (
        f"You are assisting a healthcare researcher. They have asked: '{query}'. "
        f"Based on their query, here are some AI researchers that might help:\n\n"
        f"{profiles_text}\n"
        f"Please provide further assistance or answer any questions they might have."
    )

    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate response (adjust parameters as needed)
    output = model.generate(
        input_ids,
        max_length=500,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )

    # Decode output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Step 7: Main interaction loop
def main():
    print("Welcome to the AI Researcher Matching System!")
    while True:
        query = input("\nPlease enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        relevant_profiles = find_relevant_profiles(query)
        response = generate_response(query, relevant_profiles)
        print("\n" + response)

if __name__ == "__main__":
    main()
