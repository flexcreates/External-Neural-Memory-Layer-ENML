import json
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict

from .config import GRAPH_DIR
from .logger import get_logger

logger = get_logger(__name__)

class MemoryFeedbackSystem:
    """Tracks which memories are retrieved and used to prune weak facts over time."""
    
    def __init__(self):
        self.stats_path = GRAPH_DIR / "feedback_stats.json"
        
        # fact_id -> { retrieved_count: int, used_in_response: int, user_corrected: bool }
        self.retrieval_stats = defaultdict(lambda: {
            'retrieved_count': 0,
            'used_in_response': 0,
            'user_corrected': False
        })
        
        self._load_stats()
        
    def _load_stats(self):
        if self.stats_path.exists():
            try:
                with open(self.stats_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.retrieval_stats[k] = v
            except Exception as e:
                logger.error(f"Failed to load feedback stats: {e}")
                
    def _save_stats(self):
        try:
            with open(self.stats_path, 'w', encoding='utf-8') as f:
                json.dump(dict(self.retrieval_stats), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feedback stats: {e}")

    def log_retrieval(self, fact_id: str, was_used: bool = False):
        """Called when a memory is surfaced by Qdrant"""
        self.retrieval_stats[fact_id]['retrieved_count'] += 1
        if was_used:
            self.retrieval_stats[fact_id]['used_in_response'] += 1
        self._save_stats()

    def log_user_feedback(self, fact_id: str):
        """When user says 'no, that's wrong'"""
        self.retrieval_stats[fact_id]['user_corrected'] = True
        self._save_stats()
        
    def get_memory_quality_score(self, fact_id: str) -> float:
        """Calculates 0.0-1.0 score where low scores indicate terrible facts"""
        stats = self.retrieval_stats[fact_id]
        if stats['retrieved_count'] == 0:
            return 0.5 # Neutral baseline
            
        usefulness = stats['used_in_response'] / stats['retrieved_count']
        accuracy = 0.0 if stats['user_corrected'] else 1.0
        
        return (usefulness * 0.6) + (accuracy * 0.4)
