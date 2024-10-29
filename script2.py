import os
import sys
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import json
import re

# Step 1: Load the Llama model for text generation and similarity assessment
llama_model_path = 'meta-llama/Llama-3.2-3B'  # Replace with your model path
# Load tokenizer and model for both text generation and profile similarity checking
print("Loading tokenizer and model for Llama...")
tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
model = AutoModelForCausalLM.from_pretrained(llama_model_path)
print("Tokenizer and model loaded successfully.")

# Step 2: Load and embed AI researcher profiles
def load_profiles(profile_path='profiles.json'):
    """
    Load researcher profiles from a JSON file.
    """
    print(f"Loading profiles from {profile_path}...")
    if not os.path.exists(profile_path):
        print(f"Error: {profile_path} not found.")
        sys.exit(1)

    # Load profiles from JSON file
    with open(profile_path, 'r') as f:
        profiles = json.load(f)
    print(f"Loaded {len(profiles)} profiles.")
    return profiles

profiles = load_profiles()

# Initialize the embedding model (runs locally)
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Can be replaced with any local model
# Load the sentence transformer model for embedding generation
print(f"Loading embedding model: {embedding_model_name}...")
embedding_model = SentenceTransformer(embedding_model_name)
print("Embedding model loaded successfully.")

# Create embeddings for the profiles
print("Generating embeddings for profiles...")
profile_texts = [
    f"{p['name']}. Expertise: {p['expertise']}. {p['description']}" for p in profiles
]
# Generate embeddings for each profile
profile_embeddings = embedding_model.encode(profile_texts, convert_to_numpy=True)
print("Embeddings generated successfully.")

# Step 3: Build a FAISS index
# Get the dimension of the profile embeddings
embedding_dimension = profile_embeddings.shape[1]
print(f"Building FAISS index with embedding dimension: {embedding_dimension}...")
# Create a FAISS index to store the profile embeddings for similarity search
index = faiss.IndexFlatL2(embedding_dimension)
# Add the profile embeddings to the FAISS index
index.add(profile_embeddings)
print("FAISS index built and embeddings added.")

# Function to preprocess the query
def preprocess_query(query):
    """
    Preprocess the query by lowercasing, removing special characters, and trimming whitespace.
    """
    print(f"Preprocessing query: {query}")
    query = query.lower()  # Convert query to lowercase
    query = re.sub(r'[^a-zA-Z0-9\s]', '', query)  # Remove special characters
    query = query.strip()  # Trim whitespace
    print(f"Preprocessed query: {query}")
    return query

# Function to find relevant profiles
def find_relevant_profiles(query, top_k=1):
    # Step 4: Preprocess and generate embedding for the query
    preprocessed_query = preprocess_query(query)
    # Generate embedding for the preprocessed query
    print("Generating embedding for the query...")
    query_embedding = embedding_model.encode([preprocessed_query], convert_to_numpy=True)
    print("Query embedding generated.")

    # Step 5: Search in FAISS index to find the top_k most similar profiles
    print(f"Searching for top {top_k} relevant profiles...")
    distances, indices = index.search(query_embedding, top_k)
    print(f"Initial search distances: {distances}")

    # If the closest match is not very similar, expand the search
    if distances[0][0] > 1.0:  # Threshold for similarity, can be adjusted
        print("Query is vague, expanding search criteria...")
        # Expand the number of profiles to return if the initial match is not good enough
        top_k = min(len(profiles), top_k * 2)
        distances, indices = index.search(query_embedding, top_k)
        print(f"Expanded search distances: {distances}")

    # Retrieve the corresponding profiles based on the indices from the FAISS search
    relevant_profiles = [profiles[idx] for idx in indices[0]]
    print(f"Found {len(relevant_profiles)} relevant profiles.")

    # Step 6: Check similarity with AI agent
    relevant_profiles = filter_similar_profiles_with_ai_agent(query, relevant_profiles)
    
    return relevant_profiles

# Function to filter similar profiles using the AI agent
def filter_similar_profiles_with_ai_agent(query, relevant_profiles):
    """
    Use the AI agent to judge if the returned profiles are similar enough to the query.
    """
    print("Filtering similar profiles using AI agent...")
    filtered_profiles = []

    for profile in relevant_profiles:
        # Build a text representation of the profile for evaluation
        profile_text = f"Name: {profile['name']}. Expertise: {profile['expertise']}. Description: {profile['description']}"
        # Create a prompt to ask the AI agent whether the profile matches the query
        prompt = (
            f"You are an AI agent. A user has asked the following query: '{query}'.\n"
            f"The system has found the following profile:\n{profile_text}\n"
            f"Does this profile closely match the user's query? Answer 'yes' or 'no'."
        )

        # Tokenize input for the model
        print(f"Evaluating profile: {profile['name']}")
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # Generate response from the model
        output = model.generate(
            input_ids,
            max_length=1000,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )

        # Decode the response to check if the profile is a good match
        response = tokenizer.decode(output[0], skip_special_tokens=True).strip().lower()
        print(f"AI agent response for profile '{profile['name']}': {response}")
        if 'yes' in response:
            filtered_profiles.append(profile)

    print(f"Filtered profiles count: {len(filtered_profiles)}")
    return filtered_profiles

# Function to generate a response
def generate_response(query, relevant_profiles):
    # Build a prompt that includes the profiles to generate a response for the user
    profiles_text = '\n'.join([
        f"Name: {p['name']}\nExpertise: {p['expertise']}\nDescription: {p['description']}\n"
        for p in relevant_profiles
    ])
    prompt = (
        f"You are assisting a healthcare researcher. They have asked: '{query}'. "
        f"Based on their query, here are some AI researchers that might help:\n\n"
        f"{profiles_text}\n"
        f"Please provide further assistance or answer any questions they might have."
    )

    # Tokenize input for the text generation model
    print("Generating response for the user...")
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate response (adjust parameters as needed)
    output = model.generate(
        input_ids,
        max_length=5000,  # Limit the length of the generated response
        num_return_sequences=1,  # Generate only one response
        no_repeat_ngram_size=2,  # Avoid repeating phrases
        early_stopping=True,  # Stop early if the response is complete
    )

    # Decode the generated response to return to the user
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Response generated successfully.")
    return response

# Step 7: Main interaction loop
def main():
    print("Welcome to the AI Researcher Matching System!")
    while True:
        # Prompt user for a query
        query = input("Please enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            # Exit the loop if the user types 'exit'
            print("Goodbye!")
            break

        # Find relevant profiles based on the query
        print("Finding relevant profiles...")
        relevant_profiles = find_relevant_profiles(query, top_k=3)  # Retrieve top 3 profiles
        # Generate a response using the relevant profiles
        response = generate_response(query, relevant_profiles)
        # Print the generated response
        print("" + response)

if __name__ == "__main__":
    main()
