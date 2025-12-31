"""
Main LangGraph agent for file organization.
"""
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition

from app.agents.state import AgentState, AgentPhase, ObservationState, ReasoningState, DecisionState, ActionState, MemoryState
from app.agents.nodes import AgentNodes
from app.database.session import SessionLocal
from app.utils.logging import get_logger

logger = get_logger(__name__)


class FileOrganizerAgent:
    """Main agent class using LangGraph."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.graph = self._build_graph()
        self.checkpointer = MemorySaver()
        logger.info("FileOrganizerAgent initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        # Create nodes
        db_session = SessionLocal()
        nodes = AgentNodes(db_session)
        
        # Initialize graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("observe", nodes.observe_node)
        workflow.add_node("reason", nodes.reason_node)
        workflow.add_node("decide", nodes.decide_node)
        workflow.add_node("act", nodes.act_node)
        workflow.add_node("remember", nodes.remember_node)
        workflow.add_node("complete", nodes.complete_node)
        workflow.add_node("error", nodes.error_node)
        
        # Define edges
        workflow.add_conditional_edges(
            "observe",
            self._route_after_observe,
            {
                AgentPhase.REASON.value: "reason",
                AgentPhase.ERROR.value: "error",
                AgentPhase.COMPLETE.value: "complete"
            }
        )
        
        workflow.add_conditional_edges(
            "reason",
            self._route_after_reason,
            {
                AgentPhase.DECIDE.value: "decide",
                AgentPhase.ERROR.value: "error",
                AgentPhase.COMPLETE.value: "complete"
            }
        )
        
        workflow.add_conditional_edges(
            "decide",
            self._route_after_decide,
            {
                AgentPhase.ACT.value: "act",
                AgentPhase.REASON.value: "reason",
                AgentPhase.ERROR.value: "error",
                AgentPhase.COMPLETE.value: "complete"
            }
        )
        
        workflow.add_conditional_edges(
            "act",
            self._route_after_act,
            {
                AgentPhase.REMEMBER.value: "remember",
                AgentPhase.ERROR.value: "error"
            }
        )
        
        workflow.add_conditional_edges(
            "remember",
            self._route_after_remember,
            {
                AgentPhase.REASON.value: "reason",
                AgentPhase.COMPLETE.value: "complete",
                AgentPhase.ERROR.value: "error"
            }
        )
        
        # Set entry point
        workflow.set_entry_point("observe")
        
        # Compile graph
        compiled_graph = workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["act"],  # Allow inspection before actions
            interrupt_after=["remember"]  # Allow inspection after memory
        )
        
        db_session.close()
        return compiled_graph
    
    def _route_after_observe(self, state: AgentState) -> str:
        """Route after observation."""
        if state['phase'] == AgentPhase.ERROR:
            return AgentPhase.ERROR.value
        elif not state['observation'].files_found:
            return AgentPhase.COMPLETE.value
        else:
            return AgentPhase.REASON.value
    
    def _route_after_reason(self, state: AgentState) -> str:
        """Route after reasoning."""
        if state['phase'] == AgentPhase.ERROR:
            return AgentPhase.ERROR.value
        elif not state['observation'].files_found and not state['reasoning'].current_file:
            return AgentPhase.COMPLETE.value
        else:
            return AgentPhase.DECIDE.value
    
    def _route_after_decide(self, state: AgentState) -> str:
        """Route after decision."""
        if state['phase'] == AgentPhase.ERROR:
            return AgentPhase.ERROR.value
        elif state['decision'].should_skip:
            if state['observation'].files_found:
                return AgentPhase.REASON.value
            else:
                return AgentPhase.COMPLETE.value
        else:
            return AgentPhase.ACT.value
    
    def _route_after_act(self, state: AgentState) -> str:
        """Route after action."""
        if state['phase'] == AgentPhase.ERROR:
            return AgentPhase.ERROR.value
        else:
            return AgentPhase.REMEMBER.value
    
    def _route_after_remember(self, state: AgentState) -> str:
        """Route after memory."""
        if state['phase'] == AgentPhase.ERROR:
            return AgentPhase.ERROR.value
        elif state['observation'].files_found:
            return AgentPhase.REASON.value
        else:
            return AgentPhase.COMPLETE.value
    
    def create_initial_state(self, directory: str, session_config: Dict[str, Any]) -> AgentState:
        """Create initial agent state."""
        session_id = str(uuid.uuid4())
        
        # Merge configs
        config = {
            **self.config,
            **session_config,
            "dry_run": session_config.get("dry_run", False),
            "recursive": session_config.get("recursive", True),
            "use_llm": session_config.get("use_llm", True),
            "confidence_threshold": session_config.get("confidence_threshold", 0.7)
        }
        
        return AgentState(
            phase=AgentPhase.OBSERVE,
            session_id=session_id,
            directory=directory,
            started_at=datetime.now(),
            completed_at=None,
            
            observation=ObservationState(
                directory=directory,
                recursive=config.get("recursive", True),
                max_depth=config.get("max_depth")
            ),
            
            reasoning=ReasoningState(
                use_llm=config.get("use_llm", True),
                confidence_threshold=config.get("confidence_threshold", 0.7)
            ),
            
            decision=DecisionState(),
            action=ActionState(dry_run=config.get("dry_run", False)),
            memory=MemoryState(),
            
            files_processed=0,
            files_organized=0,
            errors=0,
            categories_used={},
            decision_sources={},
            
            messages=[],
            config=config,
            result=None,
            error=None
        )
    
    async def run(self, directory: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the agent for a directory.
        
        Args:
            directory: Directory to organize
            config: Agent configuration
            
        Returns:
            Agent execution result
        """
        config = config or {}
        
        try:
            # Create initial state
            initial_state = self.create_initial_state(directory, config)
            session_id = initial_state['session_id']
            
            logger.info(f"Starting agent session {session_id} for directory: {directory}")
            
            # Run graph
            final_state = await self.graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": session_id}}
            )
            
            # Get result
            result = final_state.get('result', {})
            result['session_id'] = session_id
            
            logger.info(f"Agent session {session_id} completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                "session_id": str(uuid.uuid4()),
                "error": str(e),
                "success": False,
                "files_processed": 0,
                "files_organized": 0,
                "errors": 1
            }
    
    async def stream(self, directory: str, config: Optional[Dict[str, Any]] = None):
        """
        Stream agent execution.
        
        Args:
            directory: Directory to organize
            config: Agent configuration
            
        Yields:
            Agent state updates
        """
        config = config or {}
        initial_state = self.create_initial_state(directory, config)
        session_id = initial_state['session_id']
        
        async for event in self.graph.astream(
            initial_state,
            config={"configurable": {"thread_id": session_id}}
        ):
            yield event