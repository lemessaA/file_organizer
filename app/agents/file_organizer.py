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
from app.database.session import get_db, engine
from sqlmodel import Session
from app.utils.logging import get_logger

logger = get_logger(__name__)


class FileOrganizerAgent:
    """Main agent class using LangGraph."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {} # Store configuration if provided, otherwise use empty dict 
        # Create a MemorySaver for checkpointing - this allows the agent to:
        # 1. Resume execution if interrupted
        # 2. Maintain conversation history across interactions
        # 3. Support multi-turn conversations
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph() # the actual graph the state machine that process files
        logger.info("FileOrganizerAgent initialized")# Log that the agent has been successfully initialized
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        # Create a database session that will be used by the nodes
        # This where file classification decisions will be stored/retrieved
        db_session = Session(engine)
        
        # Instantiate the AgentNodes class which contains all the logic functions
        nodes = AgentNodes(db_session)
        
        # Initialize graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("observe", nodes.observe_node) # step1: Scan for files
        workflow.add_node("reason", nodes.reason_node)   # step2: Classify files
        workflow.add_node("decide", nodes.decide_node)   # step3: Decide action
        workflow.add_node("act", nodes.act_node)         # step4: Execute action
        workflow.add_node("remember", nodes.remember_node) # step5: store decision
        workflow.add_node("complete", nodes.complete_node) # Final : Complete results
        workflow.add_node("error", nodes.error_node)        # Error: Handle failures
        
        # Define edges
        workflow.add_conditional_edges(
            "observe", # Starting the node 
            self._route_after_observe,
            {
                AgentPhase.REASON.value: "reason", # If reasoning needed
                AgentPhase.ERROR.value: "error", # If error occurred
                AgentPhase.COMPLETE.value: "complete" # If nothing to do
            }
        )
        # After REASONING, decide wheather to:
        # -Go to DECISION  if Classification succeeded
        # - Go to COMPLETE if all files processed
        # - Go to Error if classification failed
        
        workflow.add_conditional_edges(
            "reason",
            self._route_after_reason,
            {
                AgentPhase.DECIDE.value: "decide",
                AgentPhase.ERROR.value: "error",
                AgentPhase.COMPLETE.value: "complete"
            }
        )
        #Aftre DECISION, decide wheather to:
        # - Go to ACTION if we should move the file
        # - Go back to REASONING if we should skip this 
        # - Go to ERROR if decision failed
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
        # After ACTION, decide whether to :
        # - GO to MEMORY if action succeeded
        # - Go to ERROR if action failed
        workflow.add_conditional_edges(
            "act",
            self._route_after_act,
            {
                AgentPhase.REMEMBER.value: "remember",
                AgentPhase.ERROR.value: "error"
            }
        )
        
        # After MEMORY, decide whether to :
        # - Go back to REASONING for next file 
        # - Go to Complete if no more files
        # - Go to ERROR if memory storage failed
        workflow.add_conditional_edges(
            "remember",
            self._route_after_remember,
            {
                AgentPhase.REASON.value: "reason",
                AgentPhase.COMPLETE.value: "complete",
                AgentPhase.ERROR.value: "error"
            }
        )
        
        # Set entry point where execution begins
        workflow.set_entry_point("observe")
        
        # Compile the graph into an executable  state machine 
        # The checkpointer enables:
        # 1. State persistance across executions
        #2. Debugging by allowing inspection at specific points
        #3. Resumable workflows
        
        compiled_graph = workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["act"],  # Allow inspection before actions
            interrupt_after=["remember"]  # Allow inspection after memory
        )
        
        db_session.close()
        return compiled_graph
    
    def _route_after_observe(self, state: AgentState) -> str:
        """Route after observation (robust to missing keys)."""
        phase = state.get('phase') if isinstance(state, dict) else getattr(state, 'phase', None)
        observation = state.get('observation') if isinstance(state, dict) else getattr(state, 'observation', None)

        if phase == AgentPhase.ERROR:
            return AgentPhase.ERROR.value
        elif not observation or not getattr(observation, 'files_found', None):
            return AgentPhase.COMPLETE.value
        else:
            return AgentPhase.REASON.value
    
    def _route_after_reason(self, state: AgentState) -> str:
        """Route after reasoning (robust to missing keys)."""
        phase = state.get('phase') if isinstance(state, dict) else getattr(state, 'phase', None)
        observation = state.get('observation') if isinstance(state, dict) else getattr(state, 'observation', None)
        reasoning = state.get('reasoning') if isinstance(state, dict) else getattr(state, 'reasoning', None)

        if phase == AgentPhase.ERROR:
            return AgentPhase.ERROR.value
        elif (not observation or not getattr(observation, 'files_found', None)) and (not reasoning or not getattr(reasoning, 'current_file', None)):
            return AgentPhase.COMPLETE.value
        else:
            return AgentPhase.DECIDE.value
    
    def _route_after_decide(self, state: AgentState) -> str:
        """Route after decision (robust to missing keys)."""
        phase = state.get('phase') if isinstance(state, dict) else getattr(state, 'phase', None)
        decision = state.get('decision') if isinstance(state, dict) else getattr(state, 'decision', None)
        observation = state.get('observation') if isinstance(state, dict) else getattr(state, 'observation', None)

        if phase == AgentPhase.ERROR:
            return AgentPhase.ERROR.value
        elif decision and getattr(decision, 'should_skip', False):
            if observation and getattr(observation, 'files_found', None):
                return AgentPhase.REASON.value
            else:
                return AgentPhase.COMPLETE.value
        else:
            return AgentPhase.ACT.value
    
    def _route_after_act(self, state: AgentState) -> str:
        """Route after action (robust to missing keys)."""
        phase = state.get('phase') if isinstance(state, dict) else getattr(state, 'phase', None)

        if phase == AgentPhase.ERROR:
            return AgentPhase.ERROR.value
        else:
            return AgentPhase.REMEMBER.value
    
    def _route_after_remember(self, state: AgentState) -> str:
        """Route after memory (robust to missing keys)."""
        phase = state.get('phase') if isinstance(state, dict) else getattr(state, 'phase', None)
        observation = state.get('observation') if isinstance(state, dict) else getattr(state, 'observation', None)

        if phase == AgentPhase.ERROR:
            return AgentPhase.ERROR.value
        elif observation and getattr(observation, 'files_found', None):
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


def _build_graph(config: Optional[Dict[str, Any]] = None) -> StateGraph:
    """Module-level helper for LangGraph CLI/Studio to discover the graph.

    LangGraph may call this with a config; accept it and pass to the agent.
    """
    agent = FileOrganizerAgent(config or {})
    return agent._build_graph()