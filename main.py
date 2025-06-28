from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
import time
import gc
from typing import Optional
from contextlib import asynccontextmanager
import threading
from typing import List
from fastapi.middleware.cors import CORSMiddleware




# Lifespan management for ChromaDB client
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure directories exist
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./chroma_db", exist_ok=True)
    yield
    # Shutdown: Cleanup resources
    gc.collect()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    uploaded_files = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files. Invalid file: {file.filename}")
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            uploaded_files.append(file.filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload {file.filename}: {str(e)}")

    try:
        from embed_doc import embed_documents
        embed_documents()
        return {
            "message": f"{len(uploaded_files)} PDF files uploaded and embedded.",
            "files": uploaded_files
        }
    except Exception as e:
        # Clean up uploaded files if embedding fails
        for filename in uploaded_files:
            path = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
    
    
@app.post("/query/")
async def query_api(query: str = Form(...)):
    try:
        from query_doc import query_documents

        start_time = time.time()
        response = query_documents(query)
        end_time = time.time()

        latency = round(end_time - start_time, 3)  # in seconds

        return {
            "response": response,
            "latency_seconds": latency
        }
    except Exception as e:
        raise HTTPException(500, f"Query failed: {str(e)}")
    

@app.delete("/delete/")
async def delete_data():
    try:
        from delete_doc import delete_documents
        
        # Attempt deletion with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                delete_documents()
                return {"message": "All documents and ChromaDB deleted."}
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)  # Wait before retrying
                gc.collect()  # Force cleanup between attempts
                
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Deletion failed after {max_retries} attempts: {str(e)}"
        )