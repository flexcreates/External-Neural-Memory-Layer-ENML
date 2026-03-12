from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

class TaskStatus(str, Enum):
    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    DONE = "DONE"
    ABANDONED = "ABANDONED"

class TaskPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class CodingTask:
    title: str
    description: str
    status: TaskStatus = TaskStatus.PLANNED
    priority: TaskPriority = TaskPriority.MEDIUM
    implementation_plan: List[str] = field(default_factory=list)
    current_step_index: int = 0
    completed_steps: List[int] = field(default_factory=list)
    files_involved: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    model_used: Optional[str] = None
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "implementation_plan": self.implementation_plan,
            "current_step_index": self.current_step_index,
            "completed_steps": self.completed_steps,
            "files_involved": self.files_involved,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "notes": self.notes,
            "model_used": self.model_used,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodingTask":
        return cls(
            task_id=data.get("task_id", str(uuid.uuid4())),
            title=data.get("title", ""),
            description=data.get("description", ""),
            status=TaskStatus(data.get("status", TaskStatus.PLANNED.value)),
            priority=TaskPriority(data.get("priority", TaskPriority.MEDIUM.value)),
            implementation_plan=data.get("implementation_plan", []),
            current_step_index=data.get("current_step_index", 0),
            completed_steps=data.get("completed_steps", []),
            files_involved=data.get("files_involved", []),
            dependencies=data.get("dependencies", []),
            tags=data.get("tags", []),
            notes=data.get("notes", []),
            model_used=data.get("model_used"),
            created_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat() + "Z")
        )

@dataclass
class CodeProjectContext:
    project_id: str
    project_path: str
    language: str
    framework: str
    entry_point: str
    description: str
    last_scanned: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_path": self.project_path,
            "language": self.language,
            "framework": self.framework,
            "entry_point": self.entry_point,
            "description": self.description,
            "last_scanned": self.last_scanned
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeProjectContext":
        return cls(
            project_id=data.get("project_id", ""),
            project_path=data.get("project_path", ""),
            language=data.get("language", ""),
            framework=data.get("framework", ""),
            entry_point=data.get("entry_point", ""),
            description=data.get("description", ""),
            last_scanned=data.get("last_scanned", datetime.utcnow().isoformat() + "Z")
        )
