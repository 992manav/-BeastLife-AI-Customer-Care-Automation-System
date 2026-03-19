# Configuration Guide

## Environment Variables

### LLM Provider Setup

#### Option 1: Groq (Recommended)

**Why Groq?**

- Faster inference (5-10x faster than OpenAI)
- Cheaper ($0.10-0.27 per 1M input tokens)
- Free tier: 30k tokens/day
- Models: Mixtral-8x7b, Llama3-70b

**Setup:**

1. Go to https://console.groq.com
2. Sign up for free account
3. Create API key
4. Add to `.env`:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_api_key_here
GROQ_MODEL=mixtral-8x7b-32768  # or llama3-70b-8192
```

**Rate Limits:**

- Free: 30,000 tokens per day
- Pro: 1M tokens per day ($10/month)

#### Option 2: Google Gemini

**Why Gemini?**

- Advanced multimodal capabilities
- Good reasoning abilities
- Free tier: 60 requests/minute

**Setup:**

1. Go to https://ai.google.dev
2. Click "Get API Key"
3. Create new key
4. Add to `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSyD...your_api_key_here
GEMINI_MODEL=gemini-1.5-flash      # or gemini-1.5-pro
```

**Rate Limits:**

- Free: 60 requests/minute, 1,500 requests/day
- Paid: Based on usage billing

### Database Configuration

#### SQLite (Default - Local Development)

```env
DATABASE_URL=sqlite:///./beastlife_care.db
```

**Pros:**

- Zero setup
- File-based, easy backup
- Good for development/testing

**Cons:**

- Poor concurrency
- Limited to single machine

#### PostgreSQL (Production)

```env
DATABASE_URL=postgresql://user:password@localhost:5432/beastlife_care
```

**Setup:**

```bash
# Local installation
# macOS
brew install postgresql

# Ubuntu
sudo apt-get install postgresql

# Windows
# Download from https://www.postgresql.org/download/windows/

# Create database
createdb beastlife_care

# Create user
createuser -P beastlife
# Enter password when prompted
```

**Connection Details:**

```env
# Local
DATABASE_URL=postgresql://beastlife:password@localhost:5432/beastlife_care

# Docker
DATABASE_URL=postgresql://beastlife:password@postgres:5432/beastlife_care

# AWS RDS
DATABASE_URL=postgresql://user:pass@xxx.rds.amazonaws.com:5432/beastlife_care

# Heroku
DATABASE_URL=postgres://username:password@ec2-host:5432/dbname
```

### RAG Configuration

```env
# FAISS Index location
FAISS_INDEX_PATH=./data/faiss_index

# Documents folder
DOCS_PATH=./data/docs

# Embedding model (auto-downloaded)
# all-MiniLM-L6-v2 (default)
# all-mpnet-base-v2 (better quality)
# distiluse-base-multilingual-cased-v2 (multilingual)
```

### Logging Configuration

```env
# Log level
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Log file location
LOG_FILE=./logs/app.log

# Log format
# Default: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### API Configuration

```env
# API host and port
API_HOST=0.0.0.0            # 0.0.0.0 = listen on all interfaces
API_PORT=8000

# Environment
API_ENV=development         # development, production, staging

# CORS settings
ALLOWED_ORIGINS=*           # or specific origins: localhost:3000,example.com
```

### Streamlit Configuration

```env
# Dashboard port
STREAMLIT_PORT=8501

# Theme (light, dark, auto)
STREAMLIT_THEME_BASE=light

# Max message size
STREAMLIT_MAX_MESSAGE_SIZE=200
```

## Configuration Profiles

### Development Profile

```bash
# .env
LLM_PROVIDER=groq
GROQ_API_KEY=xxx
DATABASE_URL=sqlite:///./beastlife_care.db
LOG_LEVEL=DEBUG
API_ENV=development
```

**Usage:**

```bash
python main.py init
python main.py test "Sample query"
```

### Testing Profile

```bash
# .env.test
LLM_PROVIDER=groq
GROQ_API_KEY=xxx
DATABASE_URL=sqlite:///:memory:  # In-memory database
LOG_LEVEL=INFO
API_ENV=testing
```

**Usage:**

```bash
export ENV_FILE=.env.test
pytest
```

### Production Profile

```bash
# .env.prod
LLM_PROVIDER=groq
GROQ_API_KEY=xxx
DATABASE_URL=postgresql://user:pass@proddb:5432/beastlife
LOG_LEVEL=WARNING
API_ENV=production
API_HOST=0.0.0.0
API_PORT=8000
```

**Usage:**

```bash
docker-compose -f docker-compose.yml up -d
```

## Advanced Configuration

### Custom LLM Parameters

```python
# In src/core/llm.py, modify GenerativeModel or Groq client:

# For Gemini
model = GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=GenerationConfig(
        temperature=0.7,
        max_output_tokens=1024,
        top_p=0.9
    )
)

# For Groq
response = client.chat.completions.create(
    model="mixtral-8x7b-32768",
    messages=messages,
    temperature=0.7,
    max_tokens=1024
)
```

### Database Connection Pooling

```env
# For PostgreSQL, add these:
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600
```

### RAG Optimization

Edit `src/rag/rag.py`:

```python
# Change embedding model for better quality
embedding_model_name = "all-mpnet-base-v2"  # More accurate

# Adjust retrieval parameters
top_k = 5  # Increase for more context

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Performance Tuning

```env
# LLM Temperature (0.0-2.0)
# Lower = more deterministic, Higher = more creative
# For customer support: 0.1-0.3 (consistent)
# For brainstorming: 0.7-1.0 (diverse)

# Context length
# Groq Mixtral: up to 32K tokens
# Gemini-1.5: up to 1M tokens

# Batch size
# For embeddings: 32-128 (larger = faster but more memory)
```

## Testing Configuration

### Unit Tests

```python
# tests/conftest.py
@pytest.fixture
def test_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
```

### Integration Tests

```bash
# Use test profile
export ENV_FILE=.env.test

# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

## Troubleshooting Configuration Issues

### Issue: "Invalid LLM Provider"

```bash
# Check .env
cat .env | grep LLM_PROVIDER

# Valid values: "groq" or "gemini" (lowercase)
```

### Issue: "Database Connection Failed"

```python
# Check connection string
from src.core import get_settings
settings = get_settings()
print(settings.database_url)

# For SQLite: file should exist or will be created
# For PostgreSQL: verify host, port, credentials
```

### Issue: "FAISS Index Not Found"

```bash
# Rebuild index
python main.py init

# Check data directory
ls -la data/
ls -la data/faiss_index/
```

### Issue: "LLM API Key Invalid"

```bash
# Verify API key format
# Groq: Should start with "gsk_"
# Gemini: Should start with "AIza"

# Test connectivity
curl -H "Authorization: Bearer $GROQ_API_KEY" \
  https://api.groq.com/models
```

## Security Best Practices

### 1. Never Commit Secrets

```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo "*.log" >> .gitignore
echo "beastlife_care.db" >> .gitignore
```

### 2. Use Secure Passwords

```bash
# Generate strong password
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Use in .env
POSTGRES_PASSWORD=generated_strong_password
```

### 3. Rotate API Keys Regularly

```bash
# Monthly rotation recommended
# Update .env and restart services
docker-compose restart
```

### 4. Enable HTTPS in Production

```nginx
# Use reverse proxy (nginx)
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
    }
}
```

## Configuration for Different Scales

### Small Scale (Single Server)

```env
LLM_PROVIDER=groq
DATABASE_URL=sqlite:///./beastlife_care.db
FAISS_INDEX_PATH=./data/faiss_index
LOG_LEVEL=INFO
```

### Medium Scale (Multiple Servers)

```env
LLM_PROVIDER=groq
DATABASE_URL=postgresql://user:pass@rds-endpoint:5432/beastlife
FAISS_INDEX_PATH=/shared/faiss_index
LOG_LEVEL=WARNING
CACHE_BACKEND=redis://redis-endpoint:6379
```

### Large Scale (Enterprise)

```env
LLM_PROVIDER=groq
DATABASE_URL=postgresql://user:pass@rds-multi-az/beastlife
FAISS_INDEX_PATH=/s3://bucket/faiss_index
LOG_LEVEL=WARNING
CACHE_BACKEND=redis://elasticache-endpoint:6379
MONITORING_BACKEND=prometheus
TRACING_BACKEND=jaeger
```

---

For more examples, see `.env.example` file.
