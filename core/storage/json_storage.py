
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..logger import get_logger

logger = get_logger(__name__)

class JSONStorage:
    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def save_session(self, session_id: str, messages: List[Dict[str, Any]]) -> Path:
        """
        Saves a session to a JSON file.
        
        Args:
            session_id: The unique identifier for the session.
            messages: A list of message dictionaries (e.g., {"role": "user", "content": "..."}).
            
        Returns:
            Path to the saved file.
        """
        # Create YYYY/MM directory structure
        now = datetime.now()
        relative_path = Path(now.strftime("%Y/%m"))
        save_dir = self.sessions_dir / relative_path
        save_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = save_dir / f"{session_id}.json"
        
        session_data = {
            "session_id": session_id,
            "timestamp": now.isoformat(),
            "messages": messages
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Session {session_id} saved to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            raise

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Loads a session from a JSON file.
        Searches recursively through YYYY/MM subdirectories.
        
        Args:
            session_id: The unique identifier for the session.
            
        Returns:
            The session data dictionary, or None if not found.
        """
        # First check flat path (backward compat)
        file_path = self.sessions_dir / f"{session_id}.json"
        
        # If not found at root, search YYYY/MM subdirectories
        if not file_path.exists():
            matches = list(self.sessions_dir.rglob(f"{session_id}.json"))
            if matches:
                file_path = matches[0]
            else:
                logger.warning(f"Session '{session_id}' not found in {self.sessions_dir}.")
                return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def list_sessions(self) -> List[str]:
        """
        Returns a list of all stored session IDs (searches recursively).
        """
        return [f.stem for f in self.sessions_dir.rglob("*.json")]
