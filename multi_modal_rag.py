# multi_modal_rag.py - Multi-modal RAG with PDF, Image, Audio support
import os
import glob
from typing import List, Dict, Any
import numpy as np

# PDF processing
import PyPDF2
import pdfplumber

# Image processing
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Audio processing
import whisper

# RAG components
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

# Flask API
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

# Initialize components
print("🚀 Initializing Multi-modal RAG System...")

# 1. Embedding model
print("📚 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. ChromaDB for vector storage
print("💾 Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./multi_modal_db")
collection = chroma_client.get_or_create_collection(
    name="multi_modal_docs",
    metadata={"hnsw:space": "cosine"}
)

# 3. Groq LLM
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("⚠️ OPENAI_API_KEY environment variable not set!")
    print("Please set it using: $env:OPENAI_API_KEY = 'your-key'")
    api_key = input("Or enter your Groq API key now: ")
    os.environ['OPENAI_API_KEY'] = api_key

groq_client = Groq(api_key=api_key)

# 4. Image captioning model (BLIP)
print("📸 Loading image captioning model (BLIP)...")
try:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("✅ BLIP model loaded")
except Exception as e:
    print(f"⚠️ Error loading BLIP: {e}")
    print("   Image captioning will be disabled")
    blip_processor = None
    blip_model = None

# 5. Audio transcription model (Whisper)
print("🎵 Loading audio transcription model (Whisper)...")
try:
    whisper_model = whisper.load_model("base")
    print("✅ Whisper model loaded")
except Exception as e:
    print(f"⚠️ Error loading Whisper: {e}")
    print("   Audio transcription will be disabled")
    whisper_model = None

print("✅ All components initialized!\n")

# ============ PDF PROCESSING ============
def process_pdf(file_path: str) -> List[str]:
    """Extract text from PDF file"""
    print(f"📄 Processing PDF: {file_path}")
    text_chunks = []
    
    try:
        # Try pdfplumber first (better for complex PDFs)
        with pdfplumber.open(file_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        
        if not full_text.strip():
            # Fallback to PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
        
        # Chunk the text (500 chars per chunk with overlap)
        chunk_size = 500
        overlap = 50
        for i in range(0, len(full_text), chunk_size - overlap):
            chunk = full_text[i:i + chunk_size]
            if chunk.strip():
                text_chunks.append(chunk)
                
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
    
    print(f"✅ Extracted {len(text_chunks)} chunks from PDF")
    return text_chunks

# ============ IMAGE PROCESSING ============
def process_image(file_path: str) -> str:
    """Generate caption for image using BLIP"""
    print(f"🖼️ Processing image: {file_path}")
    
    if blip_processor is None or blip_model is None:
        return f"Image file: {os.path.basename(file_path)}"
    
    try:
        image = Image.open(file_path).convert('RGB')
        inputs = blip_processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        print(f"✅ Generated caption: {caption[:100]}...")
        return caption
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return f"Unable to process image: {os.path.basename(file_path)}"

# ============ AUDIO PROCESSING ============
def process_audio(file_path: str) -> str:
    """Transcribe audio file using Whisper"""
    print(f"🎵 Processing audio: {file_path}")
    
    if whisper_model is None:
        return f"Audio file: {os.path.basename(file_path)}"
    
    try:
        result = whisper_model.transcribe(file_path)
        transcription = result["text"]
        print(f"✅ Transcription: {transcription[:100]}...")
        return transcription
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return f"Unable to transcribe audio: {os.path.basename(file_path)}"

# ============ VECTOR STORAGE ============
def add_to_vector_store(texts: List[str], source: str):
    """Add text chunks to vector database"""
    if not texts:
        print(f"⚠️ No text to add for {source}")
        return
    
    print(f"📚 Adding {len(texts)} chunks to vector store from {source}")
    
    embeddings = embedding_model.encode(texts).tolist()
    ids = [f"{source.replace('.', '_')}_{i}" for i in range(len(texts))]
    metadatas = [{"source": source, "type": source.split('.')[-1] if '.' in source else 'text'} for _ in texts]
    
    try:
        collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        print(f"✅ Added {len(texts)} chunks successfully")
    except Exception as e:
        print(f"❌ Error adding to vector store: {e}")

def process_and_store_file(file_path: str):
    """Process any file type and store in vector DB"""
    file_ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    
    print(f"\n📁 Processing file: {file_name} (Type: {file_ext})")
    
    if file_ext == '.pdf':
        chunks = process_pdf(file_path)
        add_to_vector_store(chunks, file_name)
        
    elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        caption = process_image(file_path)
        add_to_vector_store([caption], file_name)
        
    elif file_ext in ['.mp3', '.wav', '.m4a', '.ogg', '.mp4']:
        transcription = process_audio(file_path)
        add_to_vector_store([transcription], file_name)
        
    elif file_ext == '.txt':
        # Handle text files
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        add_to_vector_store([text], file_name)
        
    else:
        print(f"⚠️ Unsupported file type: {file_ext}")

# ============ QUERY & RETRIEVAL ============
def query_system(question: str, k: int = 3) -> Dict[str, Any]:
    """Query the multi-modal RAG system"""
    print(f"\n❓ Query: {question}")
    
    # Generate query embedding
    query_embedding = embedding_model.encode([question]).tolist()
    
    # Retrieve relevant chunks
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
    except Exception as e:
        print(f"❌ Error querying: {e}")
        return {
            "answer": "Error querying the database.",
            "sources": []
        }
    
    if not results['documents'] or not results['documents'][0]:
        return {
            "answer": "No relevant information found in the uploaded files.",
            "sources": []
        }
    
    # Build context from retrieved chunks
    context = "\n".join(results['documents'][0])
    sources = results['metadatas'][0]
    
    print(f"📚 Retrieved {len(results['documents'][0])} relevant chunks")
    
    # Generate answer using Groq
    prompt = f"""You are a helpful assistant answering questions based on uploaded documents (PDFs, images, audio transcriptions).

CONTEXT from uploaded files:
{context}

QUESTION: {question}

ANSWER (be concise and accurate based ONLY on the context above):"""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300
        )
        answer = response.choices[0].message.content
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        answer = f"Error generating answer: {e}"
    
    return {
        "answer": answer,
        "sources": [{"file": s['source'], "type": s['type']} for s in sources]
    }

# ============ BULK DIRECTORY PROCESSING ============
def process_directory(directory_path: str):
    """Process all supported files in a directory"""
    print(f"\n📁 Processing directory: {directory_path}")
    
    supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.mp3', '.wav', '.m4a', '.txt']
    
    for ext in supported_extensions:
        for file_path in glob.glob(f"{directory_path}/**/*{ext}", recursive=True):
            process_and_store_file(file_path)
    
    print(f"✅ Finished processing {directory_path}")

# ============ FLASK API ============
@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process a file"""
    print("\n📥 Upload request received")
    
    if 'file' not in request.files:
        print("❌ No 'file' key in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("❌ Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    print(f"✅ Received file: {file.filename}")
    
    # Save temporarily
    temp_path = f"./temp_{file.filename}"
    file.save(temp_path)
    print(f"💾 Saved to: {temp_path}")
    
    # Process and store
    process_and_store_file(temp_path)
    
    # Clean up
    os.remove(temp_path)
    print(f"🗑️ Cleaned up temp file")
    
    return jsonify({
        'message': f'File {file.filename} processed successfully',
        'filename': file.filename
    })

@app.route('/ask', methods=['POST'])
def ask():
    """Query the multi-modal RAG system"""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    result = query_system(question)
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'collection_size': collection.count(),
        'models_loaded': {
            'blip': blip_processor is not None,
            'whisper': whisper_model is not None
        }
    })

@app.route('/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    return jsonify({
        'total_documents': collection.count(),
        'supported_formats': ['PDF', 'Images (JPG, PNG, etc.)', 'Audio (MP3, WAV, etc.)', 'Text files']
    })

@app.route('/clear', methods=['POST'])
def clear_database():
    """Clear all documents from the database"""
    global collection
    print("⚠️ Clearing all documents from database...")
    
    try:
        # Delete all documents
        all_ids = collection.get()['ids']
        if all_ids:
            collection.delete(ids=all_ids)
        print(f"✅ Deleted {len(all_ids)} documents")
        return jsonify({'message': f'Cleared {len(all_ids)} documents'})
    except Exception as e:
        print(f"❌ Error clearing database: {e}")
        return jsonify({'error': str(e)}), 500

# ============ MAIN ============
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 MULTI-MODAL RAG SYSTEM READY")
    print("="*60)
    print("\nSupported file types:")
    print("  📄 PDF - Text extraction")
    print("  🖼️ Images - BLIP captioning")
    print("  🎵 Audio - Whisper transcription")
    print("  📝 Text - Direct text indexing")
    print("\nAPI Endpoints:")
    print("  POST /upload - Upload a file")
    print("  POST /ask - Query the system")
    print("  GET /health - Health check")
    print("  GET /stats - System statistics")
    print("  POST /clear - Clear all documents")
    print("\n" + "="*60)
    
    # Process a directory if specified
    import sys
    if len(sys.argv) > 1:
        process_directory(sys.argv[1])
    
    # Start Flask server
    app.run(host='0.0.0.0', port=8081, debug=True)