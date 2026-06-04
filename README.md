# 🖼️ Multi-modal RAG System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://groq.com/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)

## 📋 Overview

A **production-ready Multi-modal RAG system** that processes and answers questions from:
- 📄 **PDF documents** (text extraction)
- 🖼️ **Images** (BLIP captioning)
- 📝 **Text files** (direct indexing)

Unlike traditional RAG systems that only handle text, this system can understand the content of images and extract information from PDFs, making it ideal for document-heavy workflows.

## 🎯 Key Features

| Feature | Technology | Description |
|---------|------------|-------------|
| **PDF Processing** | pdfplumber + PyPDF2 | Extract text from PDFs with fallback support |
| **Image Captioning** | Salesforce BLIP | Generate descriptive captions for images |
| **Vector Storage** | ChromaDB | Persistent vector database with cosine similarity |
| **LLM Integration** | Groq Llama 3.1 | Fast inference for natural language answers |
| **REST API** | Flask | Upload files, query, and get responses |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Aikaksh-Singh-Routela/multi-modal-rag.git
cd multi-modal-rag

# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY="your-groq-api-key"

# Run the API
python multi_modal_rag.py