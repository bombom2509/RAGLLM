import os
import tempfile
import requests
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from uuid import uuid4
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from together import Together

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


# Text splitter setup
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Together client setup
client = Together(api_key=TOGETHER_API_KEY)

def get_together_embedding(text):
    response = client.embeddings.create(
        model="BAAI/bge-base-en-v1.5",
        input=text
    )
    return response.data[0].embedding

def process_pdf(filename, content_bytes):
    # Clear previous vectors
    try:
        index.delete(delete_all=True)
    except Exception as e:
        print(f"Skipping delete (probably no namespace yet): {e}")


    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(content_bytes)
        tmp_path = tmp_file.name

    # Extract and chunk text
    reader = PdfReader(tmp_path)
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    chunks = splitter.split_text(text)
    print(f"Extracted {len(chunks)} chunks")

    # Embed and upload in batches
    for i in range(0, len(chunks), 10):
        batch = chunks[i:i+10]
        embeddings = [get_together_embedding(chunk) for chunk in batch]
        vectors = [
            (str(uuid4()), embedding, {"text": chunk})
            for embedding, chunk in zip(embeddings, batch)
        ]
        index.upsert(vectors)

def get_rag_response(user_query):
    query_vector = get_together_embedding(user_query)
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)
    docs = [match["metadata"]["text"] for match in results["matches"]]
    context = "\n\n".join(docs)

    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant answering questions about a PDF. If the answer is not in the pdf, you will not guess."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"},
        ],
        "temperature": 0.3,
    }

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
