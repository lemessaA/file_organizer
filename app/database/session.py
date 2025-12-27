"""
Database session management using SQLModel.
"""
from typing import Generator
from sqlmodel import SQLModel, create_engine, Session
from sqlmodel.pool import QueuePool
from redis import Redis
import redis

from app.core.config import settings

# -------------------------
# SQLModel (Database) Setup
# -------------------------
# Create the database engine with connection pooling
engine = create_engine(
    settings.DATABASE_URL,      # Database URL from settings
    echo=settings.DEBUG,        # Show SQL queries if DEBUG is True
    pool_pre_ping=True,         # Check connections before using them
    pool_size=20,               # Number of connections in pool
    max_overflow=30,            # Extra connections allowed beyond pool_size
    poolclass=QueuePool         # Use QueuePool (default in SQLModel/SQLAlchemy)
)

# -------------------------
# Redis Setup
# -------------------------
redis_client: Redis = redis.from_url(
    settings.REDIS_URL,         # Redis URL from settings
    decode_responses=True,      # Decode Redis responses to string automatically
    socket_connect_timeout=5,   # Timeout in seconds for connecting to Redis
    retry_on_timeout=True       # Retry if Redis times out
)

# -------------------------
# Database Session Dependency
# -------------------------
def get_db() -> Generator[Session, None, None]:
    """
    Provide a SQLModel database session.
    Use as a FastAPI dependency:
        db: Session = Depends(get_db)
    """
    db = Session(engine)  # Open a new session bound to the engine
    try:
        yield db          # Provide session to caller
    finally:
        db.close()        # Ensure session is closed after use

# -------------------------
# Redis Dependency
# -------------------------
def get_redis() -> Redis:
    """Return the Redis client instance."""
    return redis_client

# -------------------------
# Initialize Database
# -------------------------
def init_db():
    """
    Initialize the database by creating all tables defined in SQLModel models.
    """
    from app.models.database import *  # Import all SQLModel models
    SQLModel.metadata.create_all(bind=engine)

# -------------------------
# Close Database Connections
# -------------------------
def close_db():
    """Dispose the database engine and close all connections."""
    engine.dispose()
