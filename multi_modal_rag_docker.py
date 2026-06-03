# multi_modal_rag_docker.py - Simplified Docker version
import os
import PyPDF2
import pdfplumber
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import chromadb

# Import sentence transformers (newer version works)
from sentence_transformers import SentenceTransformer

# Import transformers for BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)
CORS(app)

print("🚀 Initializing Multi-modal RAG System...")

# Initialize components
print("📚 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("💾 Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./multi_modal_db")
collection = chroma_client.get_or_create_collection(
    name="multi_modal_docs",
    metadata={"hnsw:space": "cosine"}
)

# Groq LLM
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("⚠️ OPENAI_API_KEY not set!")
    exit(1)

groq_client = Groq(api_key=api_key)

# BLIP for image captioning
print("📸 Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("✅ All models loaded!")

def process_pdf(file_path):
    """Extract text from PDF"""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if not text.strip():
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
    except Exception as e:
        print(f"PDF error: {e}")
    return text

def process_image(file_path):
    """Generate caption for image"""
    try:
        image = Image.open(file_path).convert('RGB')
        inputs = blip_processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Image error: {e}")
        return f"Image: {os.path.basename(file_path)}"

def add_to_vector_store(text, source):
    """Add text to vector database"""
    if not text or not text.strip():
        return
    embedding = embedding_model.encode([text]).tolist()
    collection.add(
        embeddings=embedding,
        documents=[text],
        ids=[source.replace('.', '_')],
        metadatas=[{"source": source}]
    )
    print(f"✅ Added {source}")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    temp_path = f"./temp_{file.filename}"
    file.save(temp_path)
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext == '.pdf':
        text = process_pdf(temp_path)
        add_to_vector_store(text, file.filename)
    elif ext in ['.jpg', '.jpeg', '.png']:
        caption = process_image(temp_path)
        add_to_vector_store(caption, file.filename)
    elif ext == '.txt':
        with open(temp_path, 'r', encoding='utf-8') as f:
            text = f.read()
        add_to_vector_store(text, file.filename)
    else:
        os.remove(temp_path)
        return jsonify({'error': f'Unsupported: {ext}'}), 400
    
    os.remove(temp_path)
    return jsonify({'message': f'File {file.filename} processed'})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'No question'}), 400
    
    # Search
    q_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=3)
    
    if not results['documents'] or not results['documents'][0]:
        return jsonify({'answer': 'No relevant information found.', 'sources': []})
    
    context = results['documents'][0][0]
    
    # Generate answer
    prompt = f"""Answer based ONLY on this context:

CONTEXT: {context}

QUESTION: {question}

ANSWER:"""
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )
    
    return jsonify({'answer': response.choices[0].message.content, 'sources': results['metadatas'][0]})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'documents': collection.count()})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 MULTI-MODAL RAG READY on port 8081")
    print("="*60)
    app.run(host='0.0.0.0', port=8081, debug=False)