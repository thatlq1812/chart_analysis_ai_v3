"""API Routes Package"""

from .documents import router as documents_router
from .health import router as health_router
from .jobs import router as jobs_router

__all__ = ["health_router", "documents_router", "jobs_router"]
