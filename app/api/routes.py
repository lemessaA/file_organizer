"""
FastAPI routes for the File Organizer Agent.
"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.api.models import (
    OrganizeRequest, OrganizeResponse, SessionStatus, SessionResult,
    MemoryStats, UpdateDecisionRequest, HealthCheck, MetricsResponse,
    AgentStatus
)
from app.agents.file_organizer import FileOrganizerAgent
from app.database.session import get_db
from app.models.database import AgentSessionDB
from app.services.memory_service import MemoryService
from app.core.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

#All endpoints in this file will  be prfixed with whatever prefix this router gets mounted with
router = APIRouter()

# Global agent instance
#This is a singleton pattern; all requests will use the same agent instance
agent = FileOrganizerAgent()
# Maps session_id -> asyncio.Task object for each running agent session
active_sessions: Dict[str, asyncio.Task] = {}

# Returns HealthCheck model (automatically serilized to JSON)
@router.get("/health", response_model=HealthCheck)
async def health_check(
    db: Session = Depends(get_db) # inject database session using dependency
):
    """Health check endpoint."""
    from app import __version__   # the application version
    
    # Check database connectivity
    db_ok = False
    try:
        db.execute("SELECT 1") # Simple query to test database connection
        db_ok = True
    except Exception:
        pass
    
    # Check OpenAI API KEY Availability
    openai_ok = bool(settings.OPENAI_API_KEY)
    
    # Check LangSmith API KEY Availability 
    langsmith_ok = bool(settings.LANGSMITH_API_KEY)
    
    # Return HealthCheck model with all status information
    return HealthCheck(
        status="healthy" if db_ok else "degraded",
        version=__version__,
        uptime=0.0,  # Would need to track startup time
        database=db_ok,
        openai=openai_ok,
        langsmith=langsmith_ok,
        timestamp=datetime.now()
    )

# Accepts OrganizeRequest model, returns OranizaResponse model
@router.post("/organize", response_model=OrganizeResponse)
async def organize_directory(
    request: OrganizeRequest,
    background_tasks: BackgroundTasks,# fastapi utility for running tasks in background
    db: Session = Depends(get_db)
):
    """Start organizing a directory."""
    # Validate directory exists
    # Convert directory path string to path object for validation
    directory = Path(request.directory)
    # then validate
    if not directory.exists():
        raise HTTPException(status_code=404, detail="Directory not found")
    # validate path is actually a directory (not file )
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    # Generate uniquie ID  for tracking this organization job
    session_id = str(uuid.uuid4())

    # Create db record for this session
    db_session = AgentSessionDB(
        session_id=session_id,
        directory=str(directory), # Convert path back to string for storage
        status=AgentStatus.PENDING,
        dry_run=request.dry_run, 
        use_llm=request.use_llm
    )
    db.add(db_session)
    db.commit()
    
    # Prepare config directory from request parameters
    config = {
        "dry_run": request.dry_run,
        "recursive": request.recursive,
        "use_llm": request.use_llm,
        "confidence_threshold": request.confidence_threshold,
        "batch_size": request.batch_size,
        "categories_filter": request.categories_filter
    }
    
    # Start agent execution to  background tasks
    background_tasks.add_task(
        run_agent_session, # Function to in background 
        session_id,
        str(directory),
        config,
        db
    )
    # log start of the session
    logger.info(f"Started organization session {session_id} for {directory}")
    
    return OrganizeResponse(
        session_id=session_id, # Generate session ID
        status=AgentStatus.PENDING,   # current status
        message="Organization started", # User-friendly messag
        estimated_time=60,  # Rough estimate
        result_url=f"/api/sessions/{session_id}/result"  # Url to check results
    )

# Background task function to run agent session

async def run_agent_session(
    session_id: str,
    directory: str,
    config: Dict[str, Any],
    db: Session
):
    """Run agent session and update database."""
    try:
        # Query database for the session record
        db_session = db.query(AgentSessionDB).filter(
            AgentSessionDB.session_id == session_id # Find session by ID
        ).first() # Getting first mathcing record
        
        if not db_session:
            logger.error(f"Session not found: {session_id}")
            return
        #Update session to RUNNING 
        db_session.status = AgentStatus.RUNNING
        db.commit()
        
        # Run Actual agent with provided directory and config
        result = await agent.run(directory, config)
        
        # Update session with result from agent execution
        db_session.status = AgentStatus.COMPLETED
        db_session.completed_at = datetime.now()
        db_session.files_processed = result.get("files_processed", 0)
        db_session.files_organized = result.get("files_organized", 0) #number of files actually organized
        db_session.errors = result.get("errors", 0)
        db_session.result = result
        
        db.commit()
        
        logger.info(f"Session {session_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Session {session_id} failed: {e}")
        
        # Update session with error
        db_session = db.query(AgentSessionDB).filter(
            AgentSessionDB.session_id == session_id
        ).first()
        
        # If session exists, update it with failure status
        if db_session:
            db_session.status = AgentStatus.FAILED
            db_session.completed_at = datetime.now()
            db_session.error_message = str(e)
            db.commit()
        
        # Remove from active sessions
        if session_id in active_sessions:
            del active_sessions[session_id]


@router.get("/sessions/{session_id}/status", response_model=SessionStatus)
async def get_session_status(
    session_id: str,
    db: Session = Depends(get_db) # Inject database session
):
    """Get status of a session."""
    db_session = db.query(AgentSessionDB).filter(
        AgentSessionDB.session_id == session_id
    ).first()
    
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Calculate progress percentage
    progress = 0.0
    if db_session.files_processed > 0 and db_session.result:
        total = db_session.result.get("total_files", db_session.files_processed)
        if total > 0:
            progress = db_session.files_processed / total
    
    return SessionStatus(
        session_id=db_session.session_id,
        status=AgentStatus(db_session.status),
        directory=db_session.directory,
        started_at=db_session.started_at,
        completed_at=db_session.completed_at,
        files_processed=db_session.files_processed,
        files_organized=db_session.files_organized,
        errors=db_session.errors,
        progress=progress,
        current_phase=None  # Would need to track in real-time
    )


@router.get("/sessions/{session_id}/result", response_model=SessionResult)
async def get_session_result(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get result of a completed session."""
    db_session = db.query(AgentSessionDB).filter(
        AgentSessionDB.session_id == session_id
    ).first()
    
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if db_session.status != AgentStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Session not completed")
    
    if not db_session.result:
        raise HTTPException(status_code=404, detail="Result not available")
    
    result = db_session.result
    
    return SessionResult(
        session_id=db_session.session_id,
        directory=db_session.directory,
        started_at=db_session.started_at,
        completed_at=db_session.completed_at,
        files_processed=result.get("files_processed", 0),
        files_organized=result.get("files_organized", 0),
        errors=result.get("errors", 0),
        categories_used=result.get("categories_used", {}),
        decision_sources=result.get("decision_sources", {}),
        dry_run=db_session.dry_run,
        classification_history=result.get("classification_history", []),
        action_history=result.get("action_history", []),
        stored_decisions=result.get("stored_decisions", [])
    )


@router.get("/sessions", response_model=List[SessionStatus])
async def list_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[AgentStatus] = None,
    db: Session = Depends(get_db)
):
    """List all sessions."""
    query = db.query(AgentSessionDB)
    
    if status:
        query = query.filter(AgentSessionDB.status == status.value)
    
    sessions = query.order_by(
        AgentSessionDB.started_at.desc()
    ).offset(skip).limit(limit).all()
    
    return [
        SessionStatus(
            session_id=s.session_id,
            status=AgentStatus(s.status),
            directory=s.directory,
            started_at=s.started_at,
            completed_at=s.completed_at,
            files_processed=s.files_processed,
            files_organized=s.files_organized,
            errors=s.errors,
            progress=0.0,
            current_phase=None
        )
        for s in sessions
    ]


@router.get("/memory/stats", response_model=MemoryStats)
async def get_memory_stats(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get memory statistics."""
    memory_service = MemoryService(db)
    stats = memory_service.get_statistics(days)
    
    return MemoryStats(**stats)


@router.put("/memory/decisions/{filename}")
async def update_decision(
    filename: str,
    request: UpdateDecisionRequest,
    db: Session = Depends(get_db)
):
    """Update a memory decision."""
    memory_service = MemoryService(db)
    
    # Validate category
    from app.models.schemas import FileCategory
    try:
        category = FileCategory(request.new_category)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category: {request.new_category}. "
                   f"Valid categories: {[c.value for c in FileCategory]}"
        )
    
    updated = memory_service.update_decision(
        filename=filename,
        new_category=category,
        new_target=request.new_target_path
    )
    
    if not updated:
        raise HTTPException(status_code=404, detail="Decision not found")
    
    return {"message": "Decision updated successfully"}


@router.delete("/memory/decisions/{filename}")
async def delete_decision(
    filename: str,
    db: Session = Depends(get_db)
):
    """Delete a memory decision."""
    memory_service = MemoryService(db)
    
    success = memory_service.forget(filename)
    
    if not success:
        raise HTTPException(status_code=404, detail="Decision not found")
    
    return {"message": "Decision deleted successfully"}


@router.get("/memory/decisions")
async def search_decisions(
    pattern: str = Query(..., description="Search pattern"),
    use_regex: bool = Query(False, description="Use regex pattern"),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Search memory decisions."""
    memory_service = MemoryService(db)
    decisions = memory_service.recall_by_pattern(pattern, use_regex)
    
    return decisions[:limit]


@router.get("/stream/{session_id}")
async def stream_session(
    session_id: str
):
    """Stream agent execution (SSE)."""
    # This would implement Server-Sent Events for real-time updates
    # For simplicity, returning placeholder
    raise HTTPException(status_code=501, detail="Streaming not implemented yet")


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    db: Session = Depends(get_db)
):
    """Get system metrics."""
    # Total sessions
    total_sessions = db.query(AgentSessionDB).count()
    
    # Active sessions
    active_sessions_count = db.query(AgentSessionDB).filter(
        AgentSessionDB.status == AgentStatus.RUNNING
    ).count()
    
    # Aggregate statistics
    from sqlalchemy import func
    
    stats = db.query(
        func.sum(AgentSessionDB.files_processed).label("total_processed"),
        func.sum(AgentSessionDB.files_organized).label("total_organized"),
        func.avg(
            func.extract('epoch', AgentSessionDB.completed_at - AgentSessionDB.started_at)
        ).label("avg_time")
    ).filter(
        AgentSessionDB.status == AgentStatus.COMPLETED
    ).first()
    
    total_processed = stats.total_processed or 0
    total_organized = stats.total_organized or 0
    avg_time = stats.avg_time or 0.0
    
    # Error rate
    error_sessions = db.query(AgentSessionDB).filter(
        AgentSessionDB.status == AgentStatus.FAILED
    ).count()
    
    error_rate = error_sessions / total_sessions if total_sessions > 0 else 0.0
    
    # Category distribution from memory
    memory_service = MemoryService(db)
    memory_stats = memory_service.get_statistics(7)
    
    return MetricsResponse(
        sessions_total=total_sessions,
        sessions_active=active_sessions_count,
        files_processed_total=total_processed,
        files_organized_total=total_organized,
        error_rate=error_rate,
        avg_processing_time=avg_time,
        category_distribution=memory_stats["category_distribution"],
        memory_usage={"database": 0.0},  # Would need actual metrics
        api_requests={"total": 0, "success": 0, "errors": 0}  # Would need tracking
    )