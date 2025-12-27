"""
FastAPI application entry point.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from prometheus_client import make_asgi_app

from app.api.routes import router
from app.core.config import settings
from app.database.session import init_db, close_db
from app.utils.logging import setup_langsmith, get_logger

logger = get_logger(__name__)

# Version
__version__ = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{__version__}")
    logger.info(f"Environment: {settings.APP_ENV}")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    # Setup LangSmith
    tracer = setup_langsmith()
    if tracer:
        app.state.langsmith_tracer = tracer
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    close_db()


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=__version__,
    description="Agentic File Organizer AI using LangGraph and FastAPI",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.DEBUG else ["localhost", "127.0.0.1"]
)

# Add Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include routers
app.include_router(router, prefix="/api", tags=["agent"])


@app.get("/")
async def root():
    """Root endpoint."""
    return JSONResponse(
        content={
            "name": settings.APP_NAME,
            "version": __version__,
            "status": "running",
            "environment": settings.APP_ENV,
            "docs": "/docs" if settings.DEBUG else None
        }
    )


@app.get("/version")
async def version():
    """Version endpoint."""
    return {"version": __version__}


@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_exception_handler(request, exc):
    """Custom 500 handler."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )