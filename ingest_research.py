import argparse
from pathlib import Path
from core.vector.retriever import Retriever
from core.config import QDRANT_RESEARCH_COLLECTION
from core.logger import get_logger

logger = get_logger(__name__)

def ingest_research(file_path: str, topic: str):
    retriever = Retriever()
    path = Path(file_path)
    
    if not path.exists():
        logger.error(f"File {file_path} not found.")
        return
        
    try:
        content = path.read_text(encoding='utf-8')
        # Very brute force chunking for brevity
        chunk_size = 1000
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            payload = {
                "type": "research",
                "source": path.name,
                "topic": topic,
                "chunk_index": i
            }
            retriever.add_memory(QDRANT_RESEARCH_COLLECTION, chunk, payload)
            
        logger.info(f"Successfully ingested {len(chunks)} chunks from {path.name}")
    except Exception as e:
        logger.error(f"Failed to ingest research: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Path to text document")
    parser.add_argument("--topic", required=True, help="Topic of the research material")
    args = parser.parse_args()
    ingest_research(args.file_path, args.topic)
