import re
import asyncio
import uuid
from typing import Dict, Any, List
from datetime import datetime
from langgraph.graph import StateGraph, END
from src.core.models import AgentState, QueryLog
from src.core.logger import setup_logger
from src.core.llm import get_llm_provider
from src.database import get_db
from src.rag import get_rag_system

logger = setup_logger(__name__)

# ================================
# NODE FUNCTIONS
# ================================

async def ingestion_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 1: Ingestion
    - Accept query input
    - Store in state
    """
    logger.info(f"Ingestion node: Processing query")
    state["metadata"] = {
        "node_start_time": datetime.utcnow().isoformat(),
        "query_id": str(uuid.uuid4())
    }
    return state

async def preprocessing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 2: Preprocessing
    - Clean text
    - Remove PII using regex
    - Normalize text
    """
    logger.info("Preprocessing node: Cleaning and normalizing query")
    
    query = state["query"]
    
    # Remove extra whitespace
    sanitized = " ".join(query.split())
    
    # Remove PII patterns
    # Email
    sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', sanitized)
    
    # Phone numbers
    sanitized = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', sanitized)
    
    # Credit card-like numbers
    sanitized = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD]', sanitized)
    
    # Social security numbers
    sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', sanitized)
    
    # Normalize case
    sanitized_lower = sanitized.lower()
    
    state["sanitized_query"] = sanitized_lower
    logger.debug(f"Sanitized query: {sanitized_lower}")
    
    return state

async def parallel_ai_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 3: Parallel AI Processing
    - Run classify_query, extract_entities, analyze_sentiment concurrently
    - Merge results
    """
    logger.info("Parallel AI node: Running concurrent AI tasks")
    
    llm_provider = get_llm_provider()
    query = state["sanitized_query"]
    
    # Define parallel tasks
    async def classify_query():
        """Classify query into categories."""
        categories = [
            "order_tracking",
            "delivery_delays",
            "refund_requests",
            "product_complaints",
            "subscription_issues",
            "payment_issues",
            "general_questions",
            "other"
        ]
        result = await llm_provider.classify(query, categories)
        logger.debug(f"Classification result: {result}")
        return result
    
    async def extract_entities():
        """Extract entities from query."""
        result = await llm_provider.extract_entities(query)
        logger.debug(f"Extracted entities: {result}")
        return result
    
    async def analyze_sentiment():
        """Analyze sentiment of query."""
        result = await llm_provider.analyze_sentiment(query)
        logger.debug(f"Sentiment analysis: {result}")
        return result
    
    # Run all tasks concurrently
    try:
        classification, entities, sentiment = await asyncio.gather(
            classify_query(),
            extract_entities(),
            analyze_sentiment()
        )
        
        # Merge results into state
        raw_category = classification.get("category", "other")
        category_aliases = {
            "delivery_delay": "delivery_delays",
            "refund_request": "refund_requests",
            "product_complaint": "product_complaints",
            "subscription_issue": "subscription_issues",
            "payment_issue": "payment_issues",
            "payment_failure": "payment_issues",
            "general_question": "general_questions",
        }
        state["category"] = category_aliases.get(raw_category, raw_category)
        state["confidence"] = classification.get("confidence", 0.0)
        state["entities"] = entities
        state["sentiment"] = sentiment.get("sentiment", "neutral")
        state["all_intents"] = [
            intent.strip()
            for intent in str(classification.get("intents", [])).split(",")
        ]
        
        logger.info(
            f"AI processing complete - Category: {state['category']}, "
            f"Sentiment: {state['sentiment']}, Confidence: {state['confidence']}"
        )
    except Exception as e:
        logger.error(f"Error in parallel AI processing: {e}")
        state["category"] = "other"
        state["confidence"] = 0.0
        state["sentiment"] = "neutral"
        state["entities"] = {}
    
    return state

async def decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 4: Decision Router
    - Route to path A, B, or C based on rules
    """
    logger.info("Decision node: Routing query to appropriate path")
    
    confidence = state.get("confidence", 0.0)
    sentiment = state.get("sentiment", "neutral")
    category = state.get("category", "other")
    
    # Routing logic
    if confidence < 0.65 or sentiment == "critical":
        path = "C"  # Escalation
        logger.info("Routing to Path C: Escalation")
    elif category == "general_questions":
        path = "B"  # RAG-based response
        logger.info("Routing to Path B: RAG-based response")
    else:
        path = "A"  # API-based resolution
        logger.info("Routing to Path A: API-based resolution")
    
    state["path"] = path
    return state

async def path_a_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 5: Path A - API Resolution
    - Handle order tracking, refunds, payments
    - Support parallel API calls if multiple intents
    """
    logger.info("Path A node: API-based resolution")
    
    try:
        category = state.get("category", "other")
        entities = state.get("entities", {})
        
        # Simulate API calls based on category/intents
        responses = []
        
        if category == "order_tracking":
            # Simulate order tracking API
            order_id = entities.get("order_id", ["ORD-123"])[0]
            response = f"Your order {order_id} is processing. Expected delivery in 3-5 business days."
            responses.append(response)
        
        elif category == "refund_request":
            # Simulate refund API
            response = "We've received your refund request. You'll receive a confirmation email within 24 hours."
            responses.append(response)
        
        elif category == "payment_issue":
            # Simulate payment API
            response = "We've identified a payment issue. Please update your payment method in account settings."
            responses.append(response)
        
        else:
            # Generic API resolution
            response = f"Processing your {category} request. Our system will handle this automatically."
            responses.append(response)
        
        state["response"] = " ".join(responses)
        logger.info(f"Path A response: {state['response']}")
    
    except Exception as e:
        logger.error(f"Error in Path A: {e}")
        state["response"] = "We encountered an issue processing your request. Escalating to support team."
        state["path"] = "C"  # Escalate on error
    
    return state

async def path_b_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 6: Path B - RAG-based Response
    - Load FAISS index
    - Convert query to embedding
    - Retrieve and generate answer
    """
    logger.info("Path B node: RAG-based response generation")
    
    try:
        query = state.get("query", "")
        llm_provider = get_llm_provider()
        rag_system = await get_rag_system()
        
        # Retrieve relevant documents
        retrieval_result = await rag_system.retrieve(query, top_k=3)
        
        if retrieval_result.documents:
            # Generate answer from retrieved documents
            answer = await rag_system.generate_answer(
                query,
                retrieval_result.documents,
                llm_provider
            )
            state["response"] = answer
            logger.info(f"Path B generated response using {len(retrieval_result.documents)} documents")
        else:
            state["response"] = "I don't have information about this topic. Please contact our support team."
            logger.warning("No relevant documents retrieved")
    
    except Exception as e:
        logger.error(f"Error in Path B: {e}")
        state["response"] = "We couldn't generate an answer. Escalating to support team."
        state["path"] = "C"  # Escalate on error
    
    return state

async def path_c_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 7: Path C - Escalation
    - Return escalation message
    """
    logger.info("Path C node: Escalating to support team")
    
    state["response"] = (
        "Your issue has been marked for escalation to our support team. "
        "A specialist will contact you within 24 hours."
    )
    
    return state

async def logging_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 8: Logging
    - Store results in database
    - Run asynchronously
    """
    logger.info("Logging node: Storing results")
    
    try:
        db = await get_db()
        
        # Prepare log data
        log_data = QueryLog(
            query=state.get("query", ""),
            sanitized_query=state.get("sanitized_query", ""),
            category=state.get("category", ""),
            sentiment=state.get("sentiment", ""),
            confidence=state.get("confidence", 0.0),
            response=state.get("response", ""),
            path=state.get("path", ""),
            entities=state.get("entities", {}),
            intents=state.get("all_intents", []),
            customer_id=state.get("metadata", {}).get("customer_id"),
            session_id=state.get("metadata", {}).get("session_id"),
        )
        
        # Log to database
        query_id = await db.log_query(log_data)
        state["metadata"]["query_id"] = query_id
        logger.info(f"Query logged successfully with ID: {query_id}")
    
    except Exception as e:
        logger.error(f"Error logging query: {e}")
    
    return state

# ================================
# LANGGRAPH ORCHESTRATION
# ================================

async def build_graph() -> StateGraph:
    """Build LangGraph with all nodes and edges."""
    logger.info("Building LangGraph orchestration graph")
    
    # Create state graph
    graph = StateGraph(dict)
    
    # Add nodes
    graph.add_node("ingestion", ingestion_node)
    graph.add_node("preprocessing", preprocessing_node)
    graph.add_node("parallel_ai", parallel_ai_node)
    graph.add_node("decision", decision_node)
    graph.add_node("path_a", path_a_node)
    graph.add_node("path_b", path_b_node)
    graph.add_node("path_c", path_c_node)
    graph.add_node("logging", logging_node)
    
    # Add edges - Main flow
    graph.add_edge("ingestion", "preprocessing")
    graph.add_edge("preprocessing", "parallel_ai")
    graph.add_edge("parallel_ai", "decision")
    
    # Add conditional edges - Route based on decision
    def route_path(state: Dict[str, Any]) -> str:
        path = state.get("path", "A")
        if path == "A":
            return "path_a"
        elif path == "B":
            return "path_b"
        else:
            return "path_c"
    
    graph.add_conditional_edges("decision", route_path)
    
    # All paths lead to logging and end
    graph.add_edge("path_a", "logging")
    graph.add_edge("path_b", "logging")
    graph.add_edge("path_c", "logging")
    graph.add_edge("logging", END)
    
    # Set start node
    graph.set_entry_point("ingestion")
    
    logger.info("LangGraph built successfully")
    return graph

async def execute_query(query: str, graph) -> Dict[str, Any]:
    """Execute a query through the LangGraph."""
    logger.info(f"Executing query: {query[:100]}...")
    
    start_time = datetime.utcnow()
    
    try:
        initial_state = {"query": query}
        
        # Compile and invoke graph
        runnable = graph.compile()
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            runnable.invoke,
            initial_state
        )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        result["execution_time_ms"] = execution_time
        
        logger.info(f"Query execution completed in {execution_time:.2f}ms")
        return result
    
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return {
            "query": query,
            "response": "An error occurred processing your query. Please try again.",
            "path": "C",
            "error": str(e),
        }
