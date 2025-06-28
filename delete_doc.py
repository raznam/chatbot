from fastapi import FastAPI, HTTPException
import chromadb
from chromadb.config import Settings
import os
import shutil
import time
import gc
import threading
from fastapi import FastAPI, UploadFile

app = FastAPI()

# Global client management
_chroma_client = None
_chroma_lock = threading.Lock()

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                allow_reset=True,
                persist_directory="./chroma_db",
                chroma_db_impl="duckdb+parquet"
            )
        )
    return _chroma_client

def cleanup_chroma():
    global _chroma_client
    if _chroma_client is not None:
        try:
            _chroma_client.reset()
            del _chroma_client
            _chroma_client = None
            gc.collect()
            time.sleep(1)  # Allow time for handles to release
        except Exception as e:
            print(f"Cleanup error: {e}")

@app.post("/upload")
async def upload(file: UploadFile):
    # Your upload logic
    pass

@app.post("/query")
async def query(query: str):
    # Your query logic
    pass

@app.post("/delete")
async def delete():
    with _chroma_lock:  # Prevent concurrent access
        try:
            client = get_chroma_client()
            
            # Delete collection if exists
            if "my_docs" in [col.name for col in client.list_collections()]:
                client.delete_collection("my_docs")
                client.persist()
            
            # Clean up files
            cleanup_chroma()
            
            # Delete data directory
            if os.path.exists("./data"):
                shutil.rmtree("./data", ignore_errors=True)
            
            # Delete chroma_db directory with retries
            for _ in range(5):
                try:
                    if os.path.exists("./chroma_db"):
                        shutil.rmtree("./chroma_db", ignore_errors=True)
                    break
                except Exception as e:
                    time.sleep(0.5)
            
            return {"status": "success"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))