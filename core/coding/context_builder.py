import os
from pathlib import Path
from typing import Optional
from .models import CodingTask
from ..logger import get_logger

logger = get_logger(__name__)

class CodeContextBuilder:
    def build_task_context(self, task: CodingTask) -> str:
        if not task:
            return ""
            
        lines = []
        lines.append("\n=== ACTIVE CODING TASK ===")
        lines.append(f"Title: {task.title}")
        lines.append(f"Description: {task.description}")
        lines.append(f"Status: {task.status.value}")
        
        if task.implementation_plan:
            lines.append("\nPlan:")
            for i, step in enumerate(task.implementation_plan):
                if i in task.completed_steps:
                    prefix = "[x]"
                elif i == task.current_step_index:
                    prefix = "[>]"
                else:
                    prefix = "[ ]"
                lines.append(f"  {prefix} {step}")
                
        if task.files_involved:
            lines.append(f"\nFiles Involved: {', '.join(task.files_involved)}")
            
        if task.notes:
            lines.append("\nNotes:")
            for note in task.notes:
                lines.append(f"  - {note}")
                
        lines.append("=== END TASK CONTEXT ===\n")
        return "\n".join(lines)

    def build_project_context(self, project_path: str, max_tree_lines: int = 60) -> str:
        if not project_path:
            return ""
            
        path = Path(project_path)
        if not path.exists() or not path.is_dir():
            return ""

        skip_dirs = {".git", "__pycache__", "node_modules", ".venv"}
        skip_exts = {".pyc", ".log", ".pyo", ".pyd"}
        
        tree_lines = []
        try:
            # Simple recursive walk, keeping depth track
            def walk_dir(current_path: Path, depth: int):
                if len(tree_lines) >= max_tree_lines:
                    return
                for item in sorted(current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
                    if item.name.startswith(".") and item.name != ".env.example":
                        if item.name in skip_dirs or item.is_dir():
                            continue
                    if item.is_dir() and item.name in skip_dirs:
                        continue
                    if item.is_file() and item.suffix in skip_exts:
                        continue
                        
                    indent = "  " * depth
                    if item.is_dir():
                        tree_lines.append(f"{indent}📁 {item.name}/")
                        walk_dir(item, depth + 1)
                    else:
                        tree_lines.append(f"{indent}📄 {item.name}")
                        
            walk_dir(path, 0)
            
            if len(tree_lines) >= max_tree_lines:
                tree_lines.append(f"  ... (tree truncated to {max_tree_lines} lines)")
                
        except Exception as e:
            logger.warning(f"[CodeContextBuilder] Failed to build project tree: {e}")
            return ""
            
        if not tree_lines:
            return ""
            
        return "=== PROJECT STRUCTURE ===\n" + "\n".join(tree_lines) + "\n=========================\n"

    def build_full_coding_context(self, task: Optional[CodingTask], project_path: Optional[str]) -> str:
        parts = []
        
        if task:
            parts.append(self.build_task_context(task))
            
        if project_path:
            parts.append(self.build_project_context(project_path))
            
        return "\n".join(parts)
