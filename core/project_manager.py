
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from .config import PROJECTS_DIR, ENML_ROOT
from .logger import get_logger

logger = get_logger(__name__)

class ProjectManager:
    def __init__(self):
        self.projects_root = PROJECTS_DIR
        self._ensure_root()
        
    def _ensure_root(self):
        self.projects_root.mkdir(parents=True, exist_ok=True)

    def _get_project_name(self, file_path: Path) -> str:
        """
        Infers project name from file path.
        Assumes structure: /.../Projects/<ProjectName>/...
        Or fallbacks to 'Uncategorized'.
        """
        try:
            abs_path = file_path.resolve()
            # Check if inside ENML
            if str(abs_path).startswith(str(ENML_ROOT)):
                return "ENML"
            
            # Check if inside PROJECTS_ROOT (needs to be imported or configured?)
            # For now, simplistic heuristic: parent directory name if not deep?
            # Or use explicit project registration?
            
            # Let's try to find if it is relative to a known root
            # This is tricky without strict project roots.
            # We will use the direct parent folder name as project name for now 
            # if we can't determine better.
            return abs_path.parent.name
        except Exception:
            return "Uncategorized"

    def _get_project_dir(self, project_name: str) -> Path:
        p = self.projects_root / project_name
        p.mkdir(parents=True, exist_ok=True)
        (p / "snapshots").mkdir(exist_ok=True)
        (p / "execution_logs").mkdir(exist_ok=True)
        return p

    def save_snapshot(self, file_path: str, content: str):
        """
        Saves a versioned snapshot of a file.
        """
        path = Path(file_path)
        project_name = self._get_project_name(path)
        project_dir = self._get_project_dir(project_name)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"{path.name}_v{timestamp}.py" # Assuming .py, or preserve ext
        snapshot_path = project_dir / "snapshots" / snapshot_name
        
        try:
            with open(snapshot_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update index
            self._update_file_index(project_dir, path.name, snapshot_name)
            logger.info(f"Saved snapshot {snapshot_name} for project {project_name}")
        except Exception as e:
            logger.error(f"Snapshot failed: {e}")

    def _update_file_index(self, project_dir: Path, filename: str, snapshot_name: str):
        index_path = project_dir / "file_index.json"
        index = {}
        if index_path.exists():
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    index = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not load file index {index_path}: {e}")
            
        if filename not in index:
            index[filename] = {"versions": [], "last_modified": ""}
            
        index[filename]["versions"].append(snapshot_name)
        index[filename]["last_modified"] = datetime.now().isoformat()
        
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)

    def log_execution(self, project_name: str, command: str, output: str, exit_code: int):
        project_dir = self._get_project_dir(project_name)
        log_file = project_dir / "execution_logs" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "command": command,
            "output": output,
            "exit_code": exit_code,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Logging execution failed: {e}")
