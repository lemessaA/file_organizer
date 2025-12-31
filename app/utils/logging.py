""" Logging setup for AI agent """


import logging
from typing import Optional
import os
from datetime import datetime
from app.core.config import settings


def setup_langsmith() -> Optional[str]:
    """
    Setup langsmith for observability.
    
    Returns:
        Tracer object or None if not configured
    """
    if not settings.LANGSMITH_API_KEY or not settings.LANGSMITH_TRACING:
        return None
    
    try:
        # Initialize LangSmith client
        client = Client(
            api_key=settings.LANGSMITH_API_KEY
        )
        
        # Create tracer to monitor AI agent workflows
        tracer = langChainTracer(project_name=settings.LANGSMITH_PROJECT)
        logging.info(f"Langsmith tracing enabled for project: {settings.LANGSMITH_PROJECT}")
        return tracer
    except Exception as e:
        # Log warning if Langsmith setup fails
        logging.warning(f"Failed to setup langSmith: {e}")
        return None
    
    
def get_logger(name: str) -> logging.Logger:
    """ Get a configured logger .
     - Creates a python logger with timestamp, logger name, and log level.
     - Ensures multiple handlers are not added to the same logger.
     - Sets log level based on settings.
    
    """
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

class  AgentLogger:
    """
    Custom logger for agent operations with  langsmith integration.
    
    Handles logging for different phases of agent workflow:
    - Observation: tracking files or data observed by the agent
    - Reasoning: tracking decisions or classifications
    - Action: tracking actions performed by the agent
    - Memory: tracking decisions stored in agent memory
    """

def __init__(self, session_id: str):
    self.session_id = session_id
    
    # Standard logger for the agent session 
    self.logger = get_logger(f"agent.{session_id}")
    
    # langsmith tracer (if enabled)
    self.tracer = setup_langsmith()
    
def log_observation(self, file_count: int, path:str):
    """
        Log the observation phase.

        - Logs number of files or items observed.
        - Sends data to LangSmith for tracing if enabled.
    """
    self.logger.info(f"Observed {file_count} files in {path}")
    if self.tracer:
        self.tracer.on_chain_start(
            {"name": "Observation"},
            {"file_count": file_count, "path": path},
            run_id=f"{self.session_id}_obs"
        )
        
def log_reasoning(self, filename:str, category: str, confidence: float):
    
    """ 
    Log the reasoning reasoning phase.
        - Logs classification or decision made by the agent.
        - Includes confidence score.
        - Sends data to LangSmith if enabled.
    """
    self.logger.info(f"Classified {filename} as {category}  (confidence: {confidence:.2f})")
    if self.tracer:
        self.tracer.on_chain_start(
            {"name": "Reasoning"},
            {"filename": filename, "category": category, "confidence": confidence},
            run_id=f"{self.session_id}_reason_{filename}"
            
        )
def log_action(self, action: str, source: str, target: str, success: bool):
    """
    Log the action phase.
    
    - Logs an action performed by the agent (e.g. move, copy, delete)
    - Indicates if the action succeded or failed.
    - Sends output to lansmith if enabled.
    
    """
    status = "SUCCESS" if success else "FAILED"
    
    self.logger.info(f"Action: {action} {source} -> {target} [{status}]")
    if self.tracer:
        self.tracer.on_chain_end(
            {"output": {"action": action, "success": success}},
            run_id=f"{self.session_id}_act_{source}"
        )
def log_memory(self, decision_count: int):
    """
    Log the memory phase.
    
    - Logs how many decisions were stored in agent memory.
    - Sends ouput to langsmith if enabeled  
    - 
    """
    self.logger.info(f"Stored {decision_count} decision in memory")
    if self.tracer:
        self.tracer.on_chain_end(
            {"output": {"decision-count": decision_count}},
            run_id=f"{self.session_id}_mem"
        )
    