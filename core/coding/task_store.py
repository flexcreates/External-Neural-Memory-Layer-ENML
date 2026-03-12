import json
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from .models import CodingTask, TaskStatus
from ..logger import get_logger

logger = get_logger(__name__)

class CodingTaskStore:
    def __init__(self, storage_path=None):
        if storage_path is None:
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            storage_path = os.path.join(project_root, "memory", "coding")
        self.storage_path = Path(storage_path)
        if self.storage_path.name == "tasks":
            self.tasks_dir = self.storage_path
        else:
            self.tasks_dir = self.storage_path / "tasks"
        try:
            self.tasks_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"[CodingTaskStore] Failed to create storage directory {self.tasks_dir}: {e}")

    def _get_task_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def save(self, task: CodingTask) -> bool:
        import logging
        logging.getLogger("enml.coding").debug(f"[TASK-STORE] Saving task to: {self.storage_path}")
        task.updated_at = datetime.utcnow().isoformat() + "Z"
        try:
            path = self._get_task_path(task.task_id)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(task.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"[CodingTaskStore] Failed to save task {task.task_id}: {e}")
            return False

    def load(self, task_id: str) -> Optional[CodingTask]:
        try:
            path = self._get_task_path(task_id)
            if not path.exists():
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return CodingTask.from_dict(data)
        except Exception as e:
            logger.error(f"[CodingTaskStore] Failed to load task {task_id}: {e}")
            return None

    def list(self, status: Optional[TaskStatus] = None) -> List[CodingTask]:
        tasks = []
        try:
            if not self.tasks_dir.exists():
                return tasks
                
            for file_path in self.tasks_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        task = CodingTask.from_dict(data)
                        if status is None or task.status == status:
                            tasks.append(task)
                except Exception as e:
                    logger.error(f"[CodingTaskStore] Failed to read {file_path}: {e}")
        except Exception as e:
             logger.error(f"[CodingTaskStore] Failed list tasks: {e}")
             
        # Sort by updated_at descending
        tasks.sort(key=lambda t: t.updated_at, reverse=True)
        return tasks

    def update_status(self, task_id: str, status: TaskStatus) -> bool:
        task = self.load(task_id)
        if task:
            task.status = status
            return self.save(task)
        return False

    def advance_step(self, task_id: str) -> bool:
        """Increments current_step_index and adds old index to completed_steps."""
        task = self.load(task_id)
        if task:
            old_index = task.current_step_index
            if old_index not in task.completed_steps:
                task.completed_steps.append(old_index)
            task.current_step_index += 1
            # Prevent advancing beyond the plan length
            plan_len = len(task.implementation_plan) if task.implementation_plan else 0
            if task.current_step_index > plan_len:
                 task.current_step_index = plan_len
            return self.save(task)
        return False

    def complete_step(self, task_id: str, step_index: int) -> bool:
        task = self.load(task_id)
        if task:
            if step_index not in task.completed_steps:
                 task.completed_steps.append(step_index)
            return self.save(task)
        return False

    def delete(self, task_id: str) -> bool:
        try:
            path = self._get_task_path(task_id)
            if path.exists():
                path.unlink()
                return True
        except Exception as e:
            logger.error(f"[CodingTaskStore] Failed to delete task {task_id}: {e}")
        return False

    def get_active_task(self) -> Optional[CodingTask]:
        """Returns the most recently updated IN_PROGRESS task."""
        in_progress = self.list(status=TaskStatus.IN_PROGRESS)
        if in_progress:
            return in_progress[0]
        return None

    def get_task_summary(self) -> str:
        """Returns a human-readable string of active tasks for prompt injection."""
        active_task = self.get_active_task()
        if not active_task:
            return "No active coding tasks."
            
        summary = f"Active Task: {active_task.title}\n"
        if active_task.implementation_plan:
            steps_total = len(active_task.implementation_plan)
            steps_done = len(active_task.completed_steps)
            summary += f"Progress: {steps_done}/{steps_total} steps complete.\n"
        
        return summary
