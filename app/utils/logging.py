""" Logging setup for AI agent """

import logging
from typing import Optional
import os
from datetime import datetime
from app.core.config import settings


def setup_langsmith() -> Optional[object]:
    """
    Setup LangSmith tracing if available and configured.

    Returns:
        A LangChain/LangSmith tracer instance or None if not configured/available.
    """
    if not settings.LANGSMITH_API_KEY or not settings.LANGSMITH_TRACING:
        return None

    try:
        # Import lazily to avoid hard dependency failures
        try:
            from langsmith import Client as LangSmithClient
        except Exception:
            LangSmithClient = None

        try:
            from langchain.callbacks.tracers import LangSmithTracer
            from langchain import tracing_v2
        except Exception:
            LangSmithTracer = None
            tracing_v2 = None

        if not LangSmithClient or not LangSmithTracer:
            logging.getLogger(__name__).warning("LangSmith packages not installed; tracer disabled")
            return None

        client = LangSmithClient(api_key=settings.LANGSMITH_API_KEY)
        tracer = LangSmithTracer(client=client, project=settings.LANGSMITH_PROJECT)

        # Enable LangChain v2 tracing if available
        if tracing_v2 is not None:
            try:
                tracing_v2.set_tracing_enabled(True)
            except Exception:
                # Some langchain versions expose a different API; ignore failures
                pass

        logging.getLogger(__name__).info("LangSmith tracer initialized")
        return tracer

    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to setup LangSmith: {e}")
        return None


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create a stream handler for console output 
        handler = logging.StreamHandler()
        # Set log message format 
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # set logging level (INFO, DEBUG, etc.)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    return logger


class AgentLogger:
    """
    Custom logger for agent operations.
    
    Handles logging for different phases of agent workflow:
    - Observation: tracking files or data observed by agent
    - Reasoning: tracking decisions or classifications
    - Action: tracking actions performed by agent
    - Memory: tracking decisions stored in agent memory
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        # Standard logger for agent session 
        self.logger = get_logger(f"agent.{session_id}")
        # langsmith tracer (if enabled)
        self.tracer = setup_langsmith()
    
    def log_observation(self, file_count: int, path: str):
        """Log observation phase."""
        self.logger.info(f"Observed {file_count} files in {path}")
    
    def log_reasoning(self, filename: str, category: str, confidence: float):
        """Log reasoning phase."""
        self.logger.info(f"Classified {filename} as {category} (confidence: {confidence:.2f})")
    
    def log_action(self, action: str, source: str, target: str, success: bool):
        """Log action phase."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"Action: {action} {source} -> {target} [{status}]")
    
    def log_memory(self, decision_count: int):
        """Log memory phase."""
        self.logger.info(f"Stored {decision_count} decisions in memory")
