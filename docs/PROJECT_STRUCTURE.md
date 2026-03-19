# Project Structure & Components

## 📁 Directory Structure

```
BeastLife_AI_System/
├── src/                                    # Main application source code
│   ├── __init__.py
│   ├── core/                              # Core modules
│   │   ├── __init__.py
│   │   ├── config.py                      # Configuration management
│   │   ├── logger.py                      # Logging setup
│   │   ├── models.py                      # Pydantic models
│   │   └── llm.py                         # LLM provider abstraction
│   │
│   ├── agents/                            # LangGraph orchestration
│   │   ├── __init__.py
│   │   └── orchestrator.py                # 8-node LangGraph system
│   │
│   ├── database/                          # Database layer
│   │   ├── __init__.py
│   │   └── db.py                          # SQLite/PostgreSQL support
│   │
│   ├── rag/                               # Retrieval Augmented Generation
│   │   ├── __init__.py
│   │   └── rag.py                         # FAISS + embeddings
│   │
│   ├── api/                               # FastAPI application
│   │   ├── __init__.py
│   │   └── main.py                        # API endpoints
│   │
│   └── dashboard/                         # Streamlit dashboard
│       ├── __init__.py
│       └── app.py                         # Analytics dashboard
│
├── config/                                # Configuration files
│   └── system_prompt.txt                  # AI system prompt
│
├── data/                                  # Data directory
│   ├── docs/                              # Knowledge base documents
│   └── faiss_index/                       # FAISS vector index
│
├── logs/                                  # Application logs
│
├── main.py                                # CLI entry point
├── examples.py                            # Usage examples
├── requirements.txt                       # Python dependencies
├── .env.example                           # Environment template
├── .gitignore                             # Git ignore rules
├── Dockerfile                             # Docker image
├── docker-compose.yml                     # Multi-container setup
├── README.md                              # Main documentation
├── QUICKSTART.md                          # Quick start guide
├── CONFIGURATION.md                       # Configuration guide
├── DOCKER.md                              # Docker deployment
├── DEPLOYMENT_CHECKLIST.md               # Production checklist
└── PROJECT_STRUCTURE.md                  # This file
```

## 🔧 Core Modules Overview

### config.py (Settings & Environment)

- **Purpose**: Centralized configuration management
- **Key Classes**: `Settings`
- **Features**:
  - LLM provider selection (Gemini/Groq)
  - Database URL configuration
  - API host/port settings
  - Path configuration for data and logs

### logger.py (Logging)

- **Purpose**: Structured logging with rotation
- **Key Functions**: `setup_logger()`
- **Features**:
  - Console and file output
  - Rotating file handler (10MB max)
  - Configurable log levels
  - Timestamp formatting

### models.py (Pydantic Models)

- **Purpose**: Data validation and serialization
- **Key Classes**:
  - `AgentState`: LangGraph state object
  - `QueryRequest`: API input model
  - `QueryResponse`: API output model
  - `QueryLog`: Database model
  - `Document`: RAG document model

### llm.py (LLM Provider Abstraction)

- **Purpose**: Unified interface for LLM APIs
- **Key Classes**: `LLMProvider`, `GeminiProvider`, `GroqProvider`
- **Methods**:
  - `generate()`: Text generation
  - `classify()`: Query classification
  - `extract_entities()`: Named entity extraction
  - `analyze_sentiment()`: Sentiment analysis
- **Features**:
  - Async/await support
  - Configurable models
  - JSON response parsing
  - Error handling

## 🤖 Agent Orchestration (orchestrator.py)

### 8 LangGraph Nodes:

1. **ingestion_node**
   - Accepts query input
   - Stores in state
   - Creates metadata (query ID, timestamp)

2. **preprocessing_node**
   - Text normalization
   - PII removal (email, phone, SSN, credit cards)
   - Whitespace cleaning

3. **parallel_ai_node** ⭐ **Concurrent Execution**
   - Runs 3 tasks in parallel:
     - Query classification (8 categories)
     - Entity extraction (order ID, product, issue, etc.)
     - Sentiment analysis (positive/negative/neutral/critical)
   - Uses `asyncio.gather()` for parallel execution
   - Merges results into state

4. **decision_node**
   - Routes based on confidence and sentiment:
     - Path C: Low confidence OR critical sentiment
     - Path B: General questions
     - Path A: All others

5. **path_a_node** (API Resolution)
   - Handles order tracking
   - Processes refund requests
   - Resolves payment issues
   - Simulates API calls based on category

6. **path_b_node** (RAG-Based)
   - Retrieves documents from knowledge base
   - Generates answers from context
   - RAG: Retrieval Augmented Generation

7. **path_c_node** (Escalation)
   - Escalates to human support
   - Generic escalation message
   - Marks as escalated

8. **logging_node**
   - Stores results in database
   - Async database write
   - Maintains audit trail

## 💾 Database Layer (db.py)

### Features:

- **SQLite** support (development)
- **PostgreSQL** support (production)
- Async operations
- Connection pooling
- Automatic table creation

### SQL Schema:

```sql
query_logs:
- id (primary key)
- query, sanitized_query
- category, sentiment, confidence
- response, path
- entities, intents (JSON)
- customer_id, session_id
- timestamp, resolved, escalated
- feedback_score
```

## 📚 RAG System (rag.py)

### Components:

- **SentenceTransformers**: all-MiniLM-L6-v2 (384-dim embeddings)
- **FAISS**: Vector similarity search
- **Knowledge Base**: 8 sample BeastLife documents

### Operations:

1. **Document Indexing**: Embed and store documents
2. **Query Retrieval**: Find top-k similar documents
3. **Answer Generation**: Generate from retrieved context

### Features:

- Async operations
- Index persistence
- Document metadata
- Similarity scoring
- Automatic document download

## 🌐 FastAPI Application (main.py)

### Endpoints:

| Method | Path           | Purpose                  |
| ------ | -------------- | ------------------------ |
| GET    | `/`            | API documentation        |
| GET    | `/health`      | Health check             |
| POST   | `/query`       | Process single query     |
| POST   | `/query/batch` | Process multiple queries |
| GET    | `/logs`        | Retrieve query logs      |
| GET    | `/stats`       | System statistics        |
| GET    | `/config`      | Configuration info       |

### Features:

- CORS support
- Error handling
- Request validation
- Async execution
- Health checks
- Batch processing

## 📊 Dashboard (app.py)

### Streamlit Components:

1. **Key Metrics**
   - Total queries
   - Resolved queries
   - Escalated queries
   - Resolution rate

2. **Visualizations**
   - Resolution status pie chart
   - Escalation rate bar chart
   - Query trend line chart

3. **Features**
   - Real-time refresh
   - Query log filtering
   - System health status
   - API endpoint documentation

## 🚀 CLI Entry Point (main.py)

### Commands:

```bash
python main.py init        # Initialize system
python main.py api         # Start FastAPI server
python main.py dashboard   # Start Streamlit dashboard
python main.py test "..."  # Test a query
```

## 📦 Dependencies

### Core Libraries:

- `langchain`: LLM framework
- `langgraph`: Agent orchestration
- `fastapi`: REST API framework
- `streamlit`: Dashboard framework
- `sqlalchemy`: ORM
- `faiss`: Vector search
- `sentence-transformers`: Embeddings

### LLM Clients:

- `google-generativeai`: Gemini API
- `groq`: Groq API

### Utilities:

- `pydantic`: Data validation
- `python-dotenv`: Environment management
- `aiohttp`: Async HTTP

## 🔄 Data Flow

```
User Query
    ↓
FastAPI /query endpoint
    ↓
LangGraph Orchestrator
    ↓
[8-Node Pipeline]
    ├─ Ingestion (input)
    ├─ Preprocessing (cleaning)
    ├─ Parallel AI (concurrent analysis)
    ├─ Decision (routing)
    ├─ Path A/B/C (resolution)
    ├─ Logging (storage)
    └─ Output
    ↓
API Response
    ↓
Dashboard Display
```

## 🔐 Security Features

- ✅ PII removal in preprocessing
- ✅ Environment variable management
- ✅ Async database writes
- ✅ Error message sanitization
- ✅ Request validation
- ✅ Rate limiting ready
- ✅ CORS configurable

## 📈 Scalability

- **Async/Concurrent**: All I/O operations non-blocking
- **Batch Processing**: Multiple queries in parallel
- **Database**: PostgreSQL for high concurrency
- **RAG**: FAISS for fast vector search
- **Caching**: Redis-ready for future enhancement
- **Horizontal**: Docker-ready for load balancing

## 🧪 Testing Infrastructure

- Unit test examples in `examples.py`
- Integration test pattern included
- Mock LLM responses supported
- In-memory SQLite for testing
- Load testing example provided

## 📝 Documentation Files

| File                    | Purpose                        |
| ----------------------- | ------------------------------ |
| README.md               | Complete system overview       |
| QUICKSTART.md           | 5-minute setup guide           |
| CONFIGURATION.md        | Detailed configuration options |
| DOCKER.md               | Docker deployment guide        |
| DEPLOYMENT_CHECKLIST.md | Production deployment steps    |
| PROJECT_STRUCTURE.md    | This file                      |

---

## Key Architectural Decisions

1. **LangGraph**: Chosen for declarative agent orchestration
2. **FAISS**: Fast, lightweight vector search
3. **Async/Await**: Non-blocking I/O for better throughput
4. **Dual LLM**: Flexibility to switch between providers
5. **FastAPI**: Modern, fast REST framework
6. **Streamlit**: Rapid dashboard prototyping
7. **SQLAlchemy**: ORM flexibility for multiple databases

---

**Total Components**: 18 Python modules + 6 documentation files
**Total Lines of Code**: ~3,500+
**Configuration Options**: 25+
**API Endpoints**: 7
**LangGraph Nodes**: 8
**Concurrent Tasks**: 3 (in parallel_ai_node)
