# Quick Start Guide

## 1. Setup (5 minutes)

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env with your API keys
# Choose LLM_PROVIDER: "groq" or "gemini"
```

## 2. Get API Keys

### Option A: Groq (Recommended for speed)

1. Go to https://console.groq.com
2. Sign up or login
3. Create API key
4. Copy to .env: `GROQ_API_KEY=xxx`

### Option B: Google Gemini

1. Go to https://ai.google.dev
2. Get API key from Google AI Studio
3. Copy to .env: `GEMINI_API_KEY=xxx`

## 3. Initialize System

```bash
python main.py init
```

Expected output:

```
✓ LLM Provider validated: groq
✓ Database initialized
✓ RAG system initialized
✓ LangGraph built
✅ System initialization complete!
```

## 4. Run Components

### Terminal 1: Start API Server

```bash
python main.py api
```

Server will start on `http://localhost:8000`

- API Docs: http://localhost:8000/docs
- OpenAPI: http://localhost:8000/openapi.json

### Terminal 2: Start Dashboard

```bash
python main.py dashboard
```

Dashboard will open on `http://localhost:8501`

### Terminal 3: Test System

```bash
python main.py test "Track my order"
python main.py test "What membership plans do you offer?"
python main.py test "I need a refund immediately!"
```

## 5. Test API Endpoints

### Process Single Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I track my order?",
    "customer_id": "CUST-123"
  }'
```

### Batch Process

```bash
curl -X POST http://localhost:8000/query/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"query": "Track my order"},
    {"query": "I need a refund"}
  ]'
```

### Get Statistics

```bash
curl http://localhost:8000/stats
```

### Get Logs

```bash
curl http://localhost:8000/logs?limit=10
```

## 6. Dashboard Usage

1. Open http://localhost:8501
2. View real-time metrics:
   - Total queries processed
   - Resolution rate
   - Escalation count
3. View query trends
4. Monitor system health

## 7. Common Issues

### "API key not set"

```bash
# Verify .env has GROQ_API_KEY or GEMINI_API_KEY
cat .env | grep API_KEY

# Or set in terminal
export GROQ_API_KEY=...
```

### "Connection refused" on localhost

```bash
# Check if port 8000 is in use
# Windows: netstat -ano | findstr :8000
# Linux/Mac: lsof -i :8000

# Kill process on port 8000
# Windows: taskkill /PID <PID> /F
```

### FAISS Index Error

```bash
# Delete cache and rebuild
rm -rf data/
python main.py init
```

## 8. Next Steps

1. **Production Deployment**
   - Use PostgreSQL instead of SQLite
   - Deploy with Docker
   - Setup monitoring with Prometheus

2. **Customize Knowledge Base**
   - Add your domain documents in `data/docs/`
   - Rebuild FAISS index: `python main.py init`

3. **Integrate with Business Systems**
   - Connect order tracking API
   - Integrate CRM system
   - Setup payment gateway

4. **Performance Tuning**
   - Increase FAISS batch size for embeddings
   - Use GPU for faster inference
   - Setup caching layer (Redis)

## 9. Example Scenarios

### Scenario 1: Customer Query

**Input:**

```json
{
  "query": "I can't log in to my account, help!",
  "customer_id": "CUST-001"
}
```

**Output:**

```json
{
  "query": "I can't log in to my account, help!",
  "category": "technical_support",
  "sentiment": "negative",
  "confidence": 0.87,
  "path": "B",
  "response": "Try resetting your password on the login page...",
  "execution_time_ms": 234.5
}
```

### Scenario 2: Refund Request

**Input:**

```json
{
  "query": "I want a refund for my purchase immediately!",
  "customer_id": "CUST-002"
}
```

**Output:**

```json
{
  "category": "refund_request",
  "sentiment": "critical",
  "confidence": 0.95,
  "path": "C",
  "response": "Your issue has been escalated to our support team..."
}
```

## 10. Performance Tips

- Use Groq for faster inference (cheaper too!)
- Process queries in batches for higher throughput
- Enable GPU for SentenceTransformers
- Use PostgreSQL for concurrent connections
- Cache frequently asked questions

## 11. Monitoring

Check system health:

```bash
curl http://localhost:8000/health
```

View logs:

```bash
tail -f logs/app.log
```

Check database:

```bash
sqlite3 beastlife_care.db "SELECT COUNT(*) FROM query_logs;"
```

## 12. Support

- **Documentation**: See README.md
- **Examples**: Run `python examples.py`
- **Logs**: Check `logs/app.log`
- **API Docs**: http://localhost:8000/docs

---

🎉 You're ready! Start building amazing customer experiences with BeastLife AI!
