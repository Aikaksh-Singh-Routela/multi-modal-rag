\# 🖼️ Multi-modal RAG System



\[!\[Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

\[!\[Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

\[!\[Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)

\[!\[Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://groq.com/)

\[!\[Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)



\## 📋 Overview



A \*\*production-ready Multi-modal RAG system\*\* that processes and answers questions from:

\- 📄 \*\*PDF documents\*\* (text extraction)

\- 🖼️ \*\*Images\*\* (BLIP captioning)

\- 📝 \*\*Text files\*\* (direct indexing)



Unlike traditional RAG systems that only handle text, this system can understand the content of images and extract information from PDFs, making it ideal for document-heavy workflows.



\## 🎯 Key Features



| Feature | Technology | Description |

|---------|------------|-------------|

| \*\*PDF Processing\*\* | pdfplumber + PyPDF2 | Extract text from PDFs with fallback support |

| \*\*Image Captioning\*\* | Salesforce BLIP | Generate descriptive captions for images |

| \*\*Vector Storage\*\* | ChromaDB | Persistent vector database with cosine similarity |

| \*\*LLM Integration\*\* | Groq Llama 3.1 | Fast inference for natural language answers |

| \*\*REST API\*\* | Flask | Upload files, query, and get responses |

| \*\*Production Ready\*\* | Docker + Gunicorn | Containerized for easy deployment |



\## 🏗️ Architecture



┌─────────────────────────────────────────────────────────────┐

│ USER REQUEST │

│ "What's in this image?" / "Summarize this PDF" │

└─────────────────────┬───────────────────────────────────────┘

↓

┌─────────────────────────────────────────────────────────────┐

│ UPLOAD ENDPOINT │

│ POST /upload │

│ ┌──────────┼──────────┼──────────┐ │

│ ↓ ↓ ↓ ↓ │

│ PDF Image Text Audio (coming soon) │

│ ↓ ↓ ↓ │

│ extract caption read │

│ ↓ ↓ ↓ │

│ ChromaDB Vector Store │

└─────────────────────────────────────────────────────────────┘

↓

┌─────────────────────────────────────────────────────────────┐

│ QUERY ENDPOINT │

│ POST /ask │

│ ↓ │

│ Retrieve relevant chunks │

│ ↓ │

│ Groq LLM generates answer │

│ ↓ │

│ RESPONSE │

└─────────────────────────────────────────────────────────────┘





\## 🛠️ Tech Stack



| Component | Technology |

|-----------|------------|

| \*\*Backend\*\* | Python 3.11, Flask |

| \*\*LLM\*\* | Groq (Llama 3.1 8B) |

| \*\*Embeddings\*\* | Sentence-Transformers (all-MiniLM-L6-v2) |

| \*\*Vector DB\*\* | ChromaDB |

| \*\*Image Captioning\*\* | Salesforce BLIP |

| \*\*PDF Processing\*\* | pdfplumber, PyPDF2 |

| \*\*Image Processing\*\* | Pillow (PIL) |

| \*\*Container\*\* | Docker, Gunicorn |



\## 📦 Installation



\### Local Development



```bash

\# Clone repository

git clone https://github.com/Aikaksh-Singh-Routela/multi-modal-rag.git

cd multi-modal-rag



\# Create virtual environment

python -m venv venv

source venv/bin/activate  # Linux/Mac

\# or

.\\venv\\Scripts\\activate  # Windows



\# Install dependencies

pip install -r requirements.txt



\# Set API key

export OPENAI\_API\_KEY="your-groq-api-key"  # Linux/Mac

\# or

$env:OPENAI\_API\_KEY="your-groq-api-key"  # Windows



\# Run the API

python multi\_modal\_rag.py

## 🔌 API Endpoints



| Method | Endpoint | Description |

|--------|----------|-------------|

| `POST` | `/upload` | Upload a file (PDF, image, or text) |

| `POST` | `/ask` | Ask a question about uploaded documents |

| `GET` | `/health` | Health check |



\### Example



```bash

\# Upload a file

curl -X POST -F "file=@document.pdf" http://localhost:8081/upload



\# Ask a question

curl -X POST http://localhost:8081/ask \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{"question": "What is this about?"}'

