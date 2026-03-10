
import os
from pathlib import Path
from typing import List, Optional
from core.config import ALLOWED_PATHS
from core.logger import get_logger

logger = get_logger(__name__)

class FileTool:
    def __init__(self):
        self.allowed_paths = [p.resolve() for p in ALLOWED_PATHS]
        
    def validate_path(self, path: str | Path) -> bool:
        """
        Checks if a path is within the allowed directories.
        """
        try:
            target = Path(path).resolve()
            for allowed in self.allowed_paths:
                if str(target).startswith(str(allowed)):
                    return True
            logger.warning(f"Access denied to path: {path}")
            return False
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return False

    def read_file(self, path: str) -> Optional[str]:
        if not self.validate_path(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Read error: {e}")
            return None

    def write_file(self, path: str, content: str) -> bool:
        if not self.validate_path(path):
            return False
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Wrote to file: {path}")
            return True
        except Exception as e:
            logger.error(f"Write error: {e}")
            return False

    def list_dir(self, path: str) -> List[str]:
        if not self.validate_path(path):
            return []
        try:
            return [str(p.name) for p in Path(path).iterdir()]
        except Exception as e:
            logger.error(f"List dir error: {e}")
            return []
