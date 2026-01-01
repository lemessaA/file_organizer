"""
Nodes for LangGraph agent.
"""
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from app.agents.state import AgentState, AgentPhase
from app.services.file_service import FileService
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.models.schemas import FileCategory, AgentAction, ClassificationResult
from app.core.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AgentNodes:
    """Collection of nodes for the LangGraph agent."""
    
    def __init__(self, db_session):
        self.file_service = FileService()
        self.llm_service = LLMService()
        self.memory_service = MemoryService(db_session)
        self.categories = list(FileCategory)
    
    def observe_node(self, state: AgentState) -> AgentState:
        """Observation node: Scan directory for files."""
        logger.info(f"Starting observation for {state['directory']}")
        
        try:
            directory = Path(state['directory'])
            recursive = state['config'].get('recursive', True)
            
            # Scan directory
            files = self.file_service.scan_directory(
                directory=directory,
                recursive=recursive,
                max_depth=state['config'].get('max_depth')
            )
            
            # Update state
            state['observation'].files_found = files
            state['observation'].total_scanned = len(files)
            state['files_processed'] = len(files)
            state['phase'] = AgentPhase.REASON
            
            logger.info(f"Observation complete: Found {len(files)} files")
            
            # Add system message
            state['messages'].append(
                SystemMessage(content=f"Observed {len(files)} files in {state['directory']}")
            )
            
        except Exception as e:
            state['phase'] = AgentPhase.ERROR
            state['error'] = f"Observation failed: {str(e)}"
            logger.error(f"Observation failed: {e}")
        
        return state
    
    def reason_node(self, state: AgentState) -> AgentState:
        """Reasoning node: Classify current file."""
        if not state['observation'].files_found:
            state['phase'] = AgentPhase.COMPLETE
            return state
        
        # Get next file
        current_file = state['observation'].files_found.pop(0)
        state['reasoning'].current_file = current_file
        
        # 1. Check memory first
        memory_decision = self.memory_service.recall(current_file.name)
        if memory_decision:
            state['reasoning'].category = memory_decision.category
            state['reasoning'].confidence = memory_decision.confidence
            state['reasoning'].decision_source = "memory"
            state['reasoning'].reasoning = "Retrieved from memory"
            logger.info(f"Memory recall for {current_file.name}: {memory_decision.category}")
        else:
            # 2. Check extension-based rules
            category = self._classify_by_extension(current_file.extension)
            if category:
                state['reasoning'].category = category
                state['reasoning'].confidence = 0.9
                state['reasoning'].decision_source = "extension"
                state['reasoning'].reasoning = f"Based on extension {current_file.extension}"
                logger.info(f"Extension classification for {current_file.name}: {category}")
            else:
                # 3. Use LLM if enabled
                if state['reasoning'].use_llm:
                    llm_result = self.llm_service.classify_by_filename(
                        filename=current_file.name,
                        extension=current_file.extension,
                        categories=[c.value for c in self.categories]
                    )
                    
                    if llm_result and llm_result.confidence >= state['reasoning'].confidence_threshold:
                        state['reasoning'].category = llm_result.category
                        state['reasoning'].confidence = llm_result.confidence
                        state['reasoning'].decision_source = "llm"
                        state['reasoning'].reasoning = llm_result.reasoning
                        logger.info(f"LLM classification for {current_file.name}: {llm_result.category}")
                    else:
                        # 4. Fallback
                        state['reasoning'].category = FileCategory.OTHER
                        state['reasoning'].confidence = 0.5
                        state['reasoning'].decision_source = "fallback"
                        state['reasoning'].reasoning = "No clear classification"
                        logger.info(f"Fallback classification for {current_file.name}: Other")
        
        # Record classification: Store classificaton for audit trial 
        classification_record = {
            "filename": current_file.name,
            "category": state['reasoning'].category.value if state['reasoning'].category else None,
            "confidence": state['reasoning'].confidence,
            "source": state['reasoning'].decision_source,
            "timestamp": datetime.now().isoformat()
        }
        state['reasoning'].classification_history.append(classification_record)
        
        # Update category statistics
        if state['reasoning'].category:
            cat_name = state['reasoning'].category.value
            state['categories_used'][cat_name] = state['categories_used'].get(cat_name, 0) + 1
        
        # Update decision source statistics
        state['decision_sources'][state['reasoning'].decision_source] = \
            state['decision_sources'].get(state['reasoning'].decision_source, 0) + 1
        
        state['phase'] = AgentPhase.DECIDE
        return state
    
    def decide_node(self, state: AgentState) -> AgentState:
        """Decision node: Decide action for current file."""
        current_file = state['reasoning'].current_file
        category = state['reasoning'].category
        
        # Check if should skip
        if (not category or 
            current_file.is_hidden or
            current_file.size > settings.MAX_FILE_SIZE):
            
            state['decision'].should_skip = True
            state['decision'].skip_reason = (
                "Hidden file" if current_file.is_hidden else
                "File too large" if current_file.size > settings.MAX_FILE_SIZE else
                "No category determined"
            )
            state['phase'] = AgentPhase.REASON if state['observation'].files_found else AgentPhase.COMPLETE
            logger.info(f"Decision: Skip {current_file.name} - {state['decision'].skip_reason}")
            return state
        
        # Create target path
        target_dir = Path(state['directory']) / category.value
        target_path = target_dir / current_file.name
        
        # Check for duplicates
        if target_path.exists():
            # Generate unique filename
            hash_obj = hashlib.md5(f"{current_file.path}{datetime.now()}".encode())
            unique_suffix = hash_obj.hexdigest()[:8]
            stem = target_path.stem
            suffix = target_path.suffix
            target_path = target_dir / f"{stem}_{unique_suffix}{suffix}"
            state['decision'].duplicate_detected = True
            state['decision'].conflict_resolution = "added_unique_suffix"
        
        # Create action
        action = AgentAction(
            action_type="move",
            source_path=current_file.path,
            target_path=str(target_path),
            category=category,
            dry_run=state['config'].get('dry_run', False),
            metadata={
                "confidence": state['reasoning'].confidence,
                "source": state['reasoning'].decision_source,
                "reasoning": state['reasoning'].reasoning
            }
        )
        
        state['decision'].action = action
        state['decision'].decisions_made.append(action.dict())
        state['phase'] = AgentPhase.ACT
        
        logger.info(f"Decision: Move {current_file.name} to {category.value}")
        return state
    
    def act_node(self, state: AgentState) -> AgentState:
        """Action node: Execute the file operation."""
        action = state['decision'].action
        
        if not action:
            state['phase'] = AgentPhase.ERROR
            state['error'] = "No action to execute"
            return state
        
        try:
            source = Path(action.source_path)
            target = Path(action.target_path)
            
            success, error = self.file_service.safe_move(
                source=source,
                target=target,
                dry_run=action.dry_run
            )
            
            state['action'].action_executed = True
            state['action'].success = success
            state['action'].error_message = error
            state['action'].source_path = str(source)
            state['action'].target_path = str(target)
            state['action'].dry_run = action.dry_run
            
            if success:
                state['files_organized'] += 1
                logger.info(f"Action successful: {source.name} -> {target}")
            else:
                state['errors'] += 1
                logger.error(f"Action failed: {error}")
            
            # Record action
            action_record = {
                "action": "move",
                "source": str(source),
                "target": str(target),
                "success": success,
                "error": error,
                "timestamp": datetime.now().isoformat(),
                "dry_run": action.dry_run
            }
            state['action'].action_history.append(action_record)
            
            state['phase'] = AgentPhase.REMEMBER
            
        except Exception as e:
            state['action'].success = False
            state['action'].error_message = str(e)
            state['errors'] += 1
            state['phase'] = AgentPhase.ERROR
            logger.error(f"Action execution failed: {e}")
        
        return state
    
    def remember_node(self, state: AgentState) -> AgentState:
        """Memory node: Store decision in memory."""
        if not state['action'].success or state['action'].dry_run:
            # Skip memory if action failed or was dry run
            state['phase'] = AgentPhase.REASON if state['observation'].files_found else AgentPhase.COMPLETE
            return state
        
        current_file = state['reasoning'].current_file
        category = state['reasoning'].category
        
        if not current_file or not category:
            state['phase'] = AgentPhase.ERROR
            return state
        
        try:
            # Store in memory
            memory_decision = self.memory_service.remember(
                filename=current_file.name,
                original_path=current_file.path,
                target_path=state['action'].target_path,
                category=category,
                decision_source=state['reasoning'].decision_source,
                confidence=state['reasoning'].confidence,
                session_id=state['session_id']
            )
            
            state['memory'].decision_stored = True
            state['memory'].memory_key = current_file.name
            state['memory'].stored_decisions.append(memory_decision.dict())
            
            logger.info(f"Memory stored for {current_file.name}")
            
        except Exception as e:
            state['memory'].decision_stored = False
            state['memory'].memory_key = None
            logger.error(f"Memory storage failed: {e}")
        
        # Move to next file or complete
        if state['observation'].files_found:
            state['phase'] = AgentPhase.REASON
        else:
            state['phase'] = AgentPhase.COMPLETE
        
        return state
    
    def complete_node(self, state: AgentState) -> AgentState:
        """Completion node: Finalize agent execution."""
        state['completed_at'] = datetime.now()
        state['phase'] = AgentPhase.COMPLETE
        
        # Compile results
        state['result'] = {
            "session_id": state['session_id'],
            "directory": state['directory'],
            "started_at": state['started_at'].isoformat(),
            "completed_at": state['completed_at'].isoformat(),
            "files_processed": state['files_processed'],
            "files_organized": state['files_organized'],
            "errors": state['errors'],
            "categories_used": state['categories_used'],
            "decision_sources": state['decision_sources'],
            "dry_run": state['config'].get('dry_run', False),
            "classification_history": state['reasoning'].classification_history,
            "action_history": state['action'].action_history,
            "stored_decisions": state['memory'].stored_decisions
        }
        
        logger.info(f"Agent completed: Processed {state['files_processed']} files, "
                   f"organized {state['files_organized']} files")
        
        return state
    
    def error_node(self, state: AgentState) -> AgentState:
        """Error handling node."""
        state['completed_at'] = datetime.now()
        state['phase'] = AgentPhase.ERROR
        
        logger.error(f"Agent failed with error: {state.get('error', 'Unknown error')}")
        
        return state
    
    def _classify_by_extension(self, extension: str) -> Optional[FileCategory]:
        """Classify file by extension using configured categories."""
        categories = settings.file_categories
        
        for category_name, extensions in categories.items():
            if extension.lower() in extensions:
                try:
                    return FileCategory(category_name)
                except ValueError:
                    continue
        
        return None