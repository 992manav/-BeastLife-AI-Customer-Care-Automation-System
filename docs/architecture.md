# BeastLife AI System - Architecture Documentation

## 📐 System Overview

BeastLife AI is a production-grade, multi-agent customer care automation system built on **LangGraph** for orchestration, **FastAPI** for API exposure, **Streamlit** for real-time monitoring, and advanced RAG with FAISS for intelligent knowledge retrieval.

### Core Objectives

- **Automated Query Resolution**: Route customer queries intelligently across multiple resolution paths
- **Scalability**: Async/concurrent execution using asyncio for high-throughput processing
- **Reliability**: Comprehensive error handling, logging, and audit trails
- **Transparency**: Real-time monitoring dashboard and detailed analytics
- **Flexibility**: Support for multiple LLM providers (Google Gemini, Groq)

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│  FastAPI       │  Streamlit Dashboard  │  CLI Interface             │
│  REST API      │  Real-time Analytics  │  Direct Execution         │
└────────┬────────────────────┬──────────────────────────────┬────────┘
         │                    │                              │
         ▼                    ▼                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  LangGraph State Machine (8-Node Orchestrator)               │   │
│  │  ┌──────────┐  ┌─────────────┐  ┌──────────────┐            │   │
│  │  │Ingestion │→ │Preprocessing│→ │ Parallel AI  │            │   │
│  │  └──────────┘  └─────────────┘  └──────────────┘            │   │
│  │       ↓               ↓                  ↓                   │   │
│  │  ┌──────────────────────────────────────────────────┐        │   │
│  │  │  Decision Node (Intelligent Routing)             │        │   │
│  │  └──────────────────────────────────────────────────┘        │   │
│  │    ↓           ↓            ↓                               │   │
│  │  Path A      Path B       Path C                           │   │
│  │  API Res.    RAG Query    Escalation                       │   │
│  │    ↓           ↓            ↓                               │   │
│  │  ┌──────────────────────────────────────────────────┐        │   │
│  │  │  Logging Node (Audit Trail & Analytics)          │        │   │
│  │  └──────────────────────────────────────────────────┘        │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────┬───────────────────┬──────────────────────┬────────────────┘
         │                   │                      │
         ▼                   ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INTEGRATION LAYER                               │
├─────────────────────────────────────────────────────────────────────┤
│  LLM Provider    │   vector Search   │   Database   │   Logging    │
│  (Gemini/Groq)   │   (FAISS + Embed) │   (SQLite/   │   (Rotation) │
│                  │   (SentTransform) │   PostgreSQL)│              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Architecture

### Query Processing Pipeline

```
1. INPUT STAGE
   │
   ├─→ FastAPI Endpoint receives QueryRequest
   │   └─→ Validation & CORS handling
   │
2. ORCHESTRATION STAGE
   │
   ├─→ Ingestion Node
   │   └─→ Store query, generate query_id, record timestamp
   │
   ├─→ Preprocessing Node
   │   ├─→ Remove extra whitespace
   │   ├─→ PII removal (email, phone, SSN patterns)
   │   ├─→ Text normalization
   │   └─→ Store sanitized_query in state
   │
   ├─→ Parallel AI Node
   │   ├─→ Classification (detect category/intent)
   │   ├─→ Entity Extraction (named entities)
   │   ├─→ Sentiment Analysis (positive/negative/neutral/critical)
   │   └─→ Confidence scoring
   │
3. DECISION STAGE
   │
   ├─→ Decision Node (Routing Logic)
   │   ├─→ confidence > 0.8 & api_available? → Path A
   │   ├─→ confidence 0.5-0.8 OR sentiment:negative? → Path B
   │   └─→ confidence < 0.5 OR sentiment:critical? → Path C
   │
4. RESOLUTION STAGE
   │
   ├─→ Path A: API Resolution
   │   ├─→ Call simulated APIs (order tracking, refunds, payments)
   │   └─→ Return structured response
   │
   ├─→ Path B: RAG Query
   │   ├─→ Convert query to embeddings (SentenceTransformers)
   │   ├─→ Search FAISS index for similar documents
   │   ├─→ Retrieve top-k matches
   │   ├─→ Call LLM with context + prompt
   │   └─→ Return augmented response
   │
   ├─→ Path C: Escalation
   │   └─→ Return escalation message with ticket info
   │
5. LOGGING STAGE
   └─→ Logging Node
       ├─→ Store QueryLog to database
       ├─→ Record metrics (latency, path, outcome)
       └─→ Trigger analytics update

OUTPUT: QueryResponse
└─→ Return to client with response, confidence, path, metadata
```

---

## 🧠 LangGraph Node Architecture

### Node States & Transitions

| Node ID | Name                | Input State               | Processing                                      | Output State                                  | Next Node(s) |
| ------- | ------------------- | ------------------------- | ----------------------------------------------- | --------------------------------------------- | ------------ |
| 1       | Ingestion           | `{query}`                 | ID generation, timestamp                        | `{query, metadata}`                           | 2            |
| 2       | Preprocessing       | `{query, metadata}`       | PII removal, normalization                      | `{query, sanitized_query, metadata}`          | 3            |
| 3       | Parallel AI         | `{sanitized_query}`       | Classification, entity extraction, sentiment    | `{category, entities, sentiment, confidence}` | 4            |
| 4       | Decision            | `{confidence, sentiment}` | Route selection based on confidence & sentiment | `{path: A/B/C}`                               | 5A/5B/5C     |
| 5A      | Path A (API)        | `{query, entities}`       | Call simulated APIs                             | `{response, path: A}`                         | 8            |
| 5B      | Path B (RAG)        | `{sanitized_query}`       | Embed query, search FAISS, generate response    | `{response, path: B}`                         | 8            |
| 5C      | Path C (Escalation) | `{query, sentiment}`      | Generate escalation message                     | `{response, path: C, ticket_id}`              | 8            |
| 8       | Logging             | `{complete state}`        | Store in DB, update metrics                     | `{response, log_id}`                          | END          |

### Shared AgentState Object

```python
{
    # Input
    "query": str,                    # Original user query

    # Preprocessing
    "sanitized_query": str,          # PII-removed, normalized query
    "metadata": {
        "query_id": str,             # Unique query identifier (UUID)
        "node_start_time": str,      # ISO timestamp
        "client_ip": str,            # Request source
        "user_id": str               # If authenticated
    }

    # Content Understanding
    "category": str,                 # e.g., "order_tracking", "refund", "payment"
    "sentiment": str,                # "positive", "negative", "neutral", "critical"
    "confidence": float,             # 0.0 to 1.0
    "entities": {
        "order_ids": [str],
        "product_names": [str],
        "amounts": [str],
        "dates": [str]
    }
    "all_intents": [str],            # Multiple detected intents

    # Resolution
    "path": str,                     # "A", "B", or "C"
    "response": str,                 # Final response to user

    # Tracking
    "log_id": str,                   # Database reference
    "execution_time": float          # Total processing time (ms)
}
```

---

## 📦 Component Architecture

### 1. Core Module (`src/core/`)

#### config.py

- **Purpose**: Centralized configuration management
- **Responsibilities**:
  - Load environment variables (.env file)
  - Validate LLM provider selection
  - Manage database connection strings
  - Set API host/port, log levels
  - Expose `Settings` singleton via `get_settings()`

#### logger.py

- **Purpose**: Structured logging with rotation
- **Features**:
  - Console + rotating file output
  - 10MB file rotation limit
  - Configurable log levels (DEBUG, INFO, WARNING, ERROR)
  - ISO timestamp formatting
  - Exposed via `setup_logger(name: str)`

#### models.py

- **Purpose**: Pydantic data models for validation
- **Key Classes**:
  - `AgentState`: LangGraph state (Dict with validation)
  - `QueryRequest`: API input model with validation
  - `QueryResponse`: API output model with metadata
  - `QueryLog`: Database ORM model for audit trail
  - `Document`: RAG document model

#### llm.py

- **Purpose**: Unified LLM provider interface
- **Providers**:
  - `GeminiProvider`: Google Gemini API integration
  - `GroqProvider`: Groq API integration (Mixtral, Llama)
- **Methods**:
  - `generate(prompt: str, temperature: float) → str`
  - `classify(query: str) → str`
  - `extract_entities(query: str) → Dict`
  - `analyze_sentiment(text: str) → str`

### 2. Orchestration Module (`src/agents/`)

#### orchestrator.py

- **Purpose**: LangGraph state machine implementation
- **Key Functions**:
  - `ingestion_node()`: Input handling + metadata
  - `preprocessing_node()`: Text cleaning + PII removal
  - `parallel_ai_node()`: Concurrent AI analysis tasks
  - `decision_node()`: Intelligent routing logic
  - `path_a_node()`: API-based resolution
  - `path_b_node()`: RAG-based resolution
  - `path_c_node()`: Escalation handling
  - `logging_node()`: Database persistence
- **Exported**:
  - `build_graph() → CompiledGraph`
  - `execute_query(query: str) → QueryResponse`

### 3. API Module (`src/api/`)

#### main.py

- **Purpose**: FastAPI REST API endpoints
- **Routes**:
  - `POST /query`: Submit customer query
  - `GET /query/{query_id}`: Retrieve query result
  - `GET /health`: Health check
  - `GET /metrics`: Aggregated analytics
- **Middleware**:
  - CORS (cross-origin resource sharing)
  - Error handling & HTTP exception conversion
- **Lifespan**:
  - `initialize_app()`: App startup (database, RAG, graph)
  - `shutdown_app()`: Cleanup on shutdown

### 4. Database Module (`src/database/`)

#### db.py

- **Purpose**: Database abstraction layer
- **Features**:
  - SQLite (default) and PostgreSQL support
  - SQLAlchemy ORM with async support
  - Connection pooling
  - Migrations support
- **Exposed Functions**:
  - `get_db() → AsyncSession`
  - `save_query_log(state: Dict)`
  - `get_query_history(user_id: str)`
  - `get_metrics()`

### 5. RAG Module (`src/rag/`)

#### rag.py

- **Purpose**: Retrieval-Augmented Generation system
- **Components**:
  - **Embedding Model**: SentenceTransformers (all-MiniLM-L6-v2)
  - **Vector Database**: FAISS (indexing) + Flat search
  - **Document Management**: Load, chunk, and index documents
- **Exposed Functions**:
  - `get_rag_system() → RAGSystem`
  - `index_documents(documents: List[str])`
  - `retrieve(query: str, top_k: int) → List[Document]`
  - `generate_response(query: str, context: List[Document]) → str`

### 6. Dashboard Module (`src/dashboard/`)

#### app.py

- **Purpose**: Streamlit real-time monitoring dashboard
- **Features**:
  - Live query metrics (throughput, latency)
  - Path distribution (A/B/C percentages)
  - Sentiment trend analysis
  - Query history and logs
  - System health status

---

## 🔌 Integration Points

### External Services

| Service               | Integration Method | Purpose                           | Config              |
| --------------------- | ------------------ | --------------------------------- | ------------------- |
| **Google Gemini**     | REST API           | Primary LLM                       | `GOOGLE_API_KEY`    |
| **Groq API**          | REST API           | Fallback/Alternative LLM          | `GROQ_API_KEY`      |
| **HuggingFace**       | Model Hub          | Embeddings (SentenceTransformers) | Auto-download       |
| **PostgreSQL/SQLite** | SQLAlchemy         | Persistent storage                | `DATABASE_URL`      |
| **FAISS**             | In-memory library  | Vector search                     | Local file indexing |

### Internal Service Integration

```
FastAPI ──┬──→ LangGraph
          └──→ RAG System

LangGraph ──┬──→ LLM Provider (Gemini/Groq)
            ├──→ Database Layer
            └──→ RAG System

RAG System ──┬──→ FAISS (Vector DB)
             ├──→ SentenceTransformers (Embeddings)
             └──→ Document Loader

Streamlit ──┬──→ Database (Metrics Query)
            └──→ API (Health Check)
```

---

## 🚀 Deployment Architecture

### Docker Composition

```yaml
Services:
  - api: FastAPI application (port 8000)
  - dashboard: Streamlit app (port 8501)
  - db: PostgreSQL database (port 5432)
  - redis: Caching layer (optional, port 6379)
```

### Environment Separation

#### Development

- SQLite database (local file)
- Debug logging (DEBUG level)
- Hot-reload enabled

#### Production

- PostgreSQL database (managed service)
- Structured JSON logging (INFO level)
- Reverse proxy (Nginx/Traefik)
- Health checks & auto-restart
- Resource limits enforced

---

## 🔐 Security Architecture

### Input Validation

- Query length limits (max 5000 chars)
- Pydantic model validation on API inputs
- Rate limiting per IP/user

### Data Protection

- PII removal in preprocessing node (regex patterns)
- Sanitized logs (no sensitive data)
- CORS policies configured
- API authentication (token-based, optional)

### Error Handling

- Try-catch blocks in all nodes
- Graceful degradation (fallback paths)
- Error logging without leaking internal details
- HTTP 500 responses on server errors

---

## 📊 Monitoring & Observability

### Logging Levels & Locations

```
logs/
├── application.log       # Combined application logs
├── error.log             # Error-level events
├── query_audit.log       # Audit trail (all query processing)
└── performance.log       # Performance metrics
```

### Key Metrics Tracked

| Metric                  | Source            | Purpose                   |
| ----------------------- | ----------------- | ------------------------- |
| Query Latency (ms)      | LangGraph nodes   | Performance monitoring    |
| Path Distribution (%)   | Logging node      | Resolution effectiveness  |
| Confidence Distribution | Decision node     | Model confidence analysis |
| Sentiment Breakdown     | Parallel AI node  | Customer emotion tracking |
| Error Rate (%)          | API error handler | System reliability        |

### Observability Integration Points

- Structured logging at every node
- Query-level tracing correlation (query_id)
- Database metrics table for aggregations
- Streamlit dashboard for real-time visualization

---

## 🔄 Scaling Strategy

### Horizontal Scaling

1. **Stateless API Layer**: FastAPI instances can scale independently
2. **Shared Database**: PostgreSQL as central data store
3. **Shared Vector DB**: FAISS index accessible to all instances
4. **Load Balancer**: Route requests across API instances

### Vertical Scaling

- Increase asyncio worker threads for concurrent query processing
- Increase FAISS batch size for vector operations
- Adjust database connection pool size

### Performance Optimizations

- Async/await throughout pipeline
- Parallel AI operations (concurrent API calls + embeddings)
- FAISS approximate nearest neighbor search (faster than exact)
- Response caching (optional Redis layer)
- Database query optimization (indexing on query_id, user_id)

---

## 📋 Summary

| Aspect             | Implementation                                |
| ------------------ | --------------------------------------------- |
| **Orchestration**  | LangGraph with 8 specialized nodes            |
| **Data Flow**      | State machine with intelligent routing        |
| **AI Integration** | Dual LLM support (Gemini/Groq) + RAG          |
| **API Framework**  | FastAPI with async/await                      |
| **Storage**        | PostgreSQL/SQLite + FAISS vectors             |
| **Monitoring**     | Structured logging + Streamlit dashboard      |
| **Deployment**     | Docker + docker-compose for production        |
| **Scalability**    | Horizontal (stateless) + async execution      |
| **Security**       | Input validation, PII removal, error handling |

---

## 📚 Related Documentation

- [README.md](README.md) - System overview and quick start
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Directory layout
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration options
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [DOCKER.md](DOCKER.md) - Docker deployment guide
- [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) - Production readiness
