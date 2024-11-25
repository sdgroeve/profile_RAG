import os
import sys
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import json
import re
import torch

# Step 1: Load the Llama model for text generation and similarity assessment
llama_model_path = 'meta-llama/Llama-3.2-1B'  # Replace with your model path
# Load tokenizer and model for both text generation and profile similarity checking
print("Loading tokenizer and model for Llama...")
tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
model = AutoModelForCausalLM.from_pretrained(llama_model_path)

pipe = pipeline(
    "text-generation", 
    model=llama_model_path, 
    max_length=500,
    truncation=True,
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

print("Tokenizer and model loaded successfully.")

# Step 2: Load and embed AI researcher profiles
def load_profiles(profile_path='researchers.json'):
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
    f"Name: {p['name']}\nUnit: {p['research_unit']}\nBio: {p['bio']}\nKeywords: {p['keywords']}\nPublications: {' '.join(p['publications'])}" for p in profiles
]

#tmp = "\n".join(profile_texts)
#with open("ugent_ai_prompt.txt","w") as f:
#    f.write(tmp)

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
    if distances[0][0] > 10.0:  # Threshold for similarity, can be adjusted
        print("Query is vague, expanding search criteria...")
        # Expand the number of profiles to return if the initial match is not good enough
        top_k = min(len(profiles), top_k * 2)
        distances, indices = index.search(query_embedding, top_k)
        print(f"Expanded search distances: {distances}")

    # Retrieve the corresponding profiles based on the indices from the FAISS search
    relevant_profiles = [profiles[idx] for idx in indices[0]]
    print(f"Found {len(relevant_profiles)} relevant profiles.")

    return relevant_profiles

# Function to generate a response
def generate_response(query, relevant_profiles):
    # Build a prompt that includes the profiles to generate a response for the user
    profiles_text = '\n'.join([
        f"Name: {p['name']}\nUnit: {['research_unit']}\Bio: {p['bio']}\Keywords: {p['keywords']}\n"
        for p in relevant_profiles
    ])
    prompt = (
        f"You are assisting a healthcare researcher. He or she created the following prompt: '{query}'. "
        f"Using this prompt a database was queried to find profiles of researchers that best match this prompt."
        f"These are the researchers that the database returned:\n\n"
        f"{profiles_text}\n"
        f"For each profile you need to decide if the profile indeed matches the query. If not then discard it."
        f"You need to return a short response that summarizes the similar profiles and points the matches between the query and the profile"
        f"Next you offer further assistance about the profiles"
        f"Suggest two questions the healthcare researcher could ask about the profiles"
    )
    print(prompt)
    print(len(prompt))
    fddd
    # Tokenize input for the text generation model
    print("Generating response for the user...")
    response = pipe(str(prompt))
    print(response)
    return
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate response (adjust parameters as needed)
    output = model.generate(
        input_ids,
        max_length=5000,  # Limit the length of the generated response
        num_return_sequences=1,  # Generate only one response
        no_repeat_ngram_size=2,  # Avoid repeating phrases
        early_stopping=True,  # Stop early if the response is complete
        do_sample=True,     # Enable sampling
        temperature=0.7,    # Adjust for randomness
        top_p=0.9,          # Nucleus sampling
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
