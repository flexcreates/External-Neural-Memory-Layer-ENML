import argparse
import json
from pathlib import Path
from core.vector.retriever import Retriever
from core.config import QDRANT_CONVERSATION_COLLECTION
from core.logger import get_logger

logger = get_logger(__name__)

def ingest_conversation(session_file: str, importance: float = 0.5):
    retriever = Retriever()
    path = Path(session_file)
    
    if not path.exists():
        logger.error(f"Session file {session_file} not found.")
        return
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
            
        messages = session_data if isinstance(session_data, list) else session_data.get("messages", [])
        transcript = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages])
        
        summary = transcript[:1500] 
        
        payload = {
            "type": "conversation",
            "session_id": path.stem,
            "importance": importance
        }
        
        retriever.add_memory(QDRANT_CONVERSATION_COLLECTION, summary, payload)
        logger.info(f"Successfully ingested conversation {path.stem}")
    except Exception as e:
        logger.error(f"Failed to ingest conversation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("session_file", help="Path to JSON session log")
    parser.add_argument("--importance", type=float, default=0.5, help="Retrieval importance")
    args = parser.parse_args()
    ingest_conversation(args.session_file, args.importance)
