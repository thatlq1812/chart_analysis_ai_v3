"""
Geo-SLM Chart Analysis API Package

Entry point: uvicorn src.api.main:app

Note: `app` is NOT imported at package level to avoid eager FastAPI
initialization when sub-modules (config, schemas, job_store) are imported
independently (e.g., in tests or other modules).
"""
