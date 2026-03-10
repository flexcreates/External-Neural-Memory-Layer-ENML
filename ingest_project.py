import argparse
from pathlib import Path
from core.vector.retriever import Retriever
from core.config import QDRANT_PROJECT_COLLECTION
from core.logger import get_logger

logger = get_logger(__name__)

def ingest_project(file_path: str, language: str = "python", module: str = "unknown"):
    retriever = Retriever()
    path = Path(file_path)
    
    if not path.exists():
        logger.error(f"File {file_path} not found.")
        return
        
    try:
        content = path.read_text(encoding='utf-8')
        # For code we could split by classes/functions, simple chunking for now
        chunk_size = 800
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            payload = {
                "type": "project",
                "file": path.name,
                "module": module,
                "language": language,
                "chunk_index": i
            }
            retriever.add_memory(QDRANT_PROJECT_COLLECTION, chunk, payload)
            
        logger.info(f"Successfully ingested {len(chunks)} project chunks from {path.name}")
    except Exception as e:
        logger.error(f"Failed to ingest project file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Path to code file")
    parser.add_argument("--language", default="python", help="Programming language")
    parser.add_argument("--module", default="unknown", help="Module or feature area")
    args = parser.parse_args()
    ingest_project(args.file_path, args.language, args.module)
