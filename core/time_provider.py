from datetime import datetime

class TimeProvider:
    """Provides a centralized and consistent time source for the ENML agent."""
    
    @staticmethod
    def now() -> datetime:
        return datetime.now()

    @staticmethod
    def formatted() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
