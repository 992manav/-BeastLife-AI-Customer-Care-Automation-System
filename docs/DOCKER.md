# Docker Deployment Guide

## Prerequisites

- Docker Desktop installed (https://www.docker.com/products/docker-desktop)
- Docker Compose installed
- API keys from Groq or Google Gemini

## Quick Start with Docker

### 1. Create .env file

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key_here
# OR
GEMINI_API_KEY=your_gemini_key_here
```

### 2. Start All Services

```bash
docker-compose up -d
```

This starts:

- **PostgreSQL**: http://localhost:5432
- **FastAPI API**: http://localhost:8000
- **Streamlit Dashboard**: http://localhost:8501
- **Redis Cache**: http://localhost:6379 (optional)

### 3. Initialize Database

```bash
docker-compose exec api python main.py init
```

### 4. Check Status

```bash
# Check all services
docker-compose ps

# View logs
docker-compose logs -f api
docker-compose logs -f dashboard

# Check API health
curl http://localhost:8000/health
```

## Individual Service Management

### Start Only API

```bash
docker-compose up -d api postgres
```

### Start Only Dashboard

```bash
docker-compose up -d dashboard api
```

### Stop All

```bash
docker-compose down
```

### Stop and Remove Volumes (Hard Reset)

```bash
docker-compose down -v
```

## Building Custom Images

### Build API Image

```bash
docker build -t beastlife-api:latest .
```

### Build Specific Service

```bash
docker-compose build --no-cache api
```

## Environment Variables Reference

```env
# LLM Configuration
LLM_PROVIDER=groq                          # groq or gemini
GROQ_API_KEY=xxx                           # Groq API key
GEMINI_API_KEY=xxx                         # Google Gemini API key

# Database
DATABASE_URL=postgresql://user:pass@postgres/beastlife_care

# API
API_HOST=0.0.0.0
API_PORT=8000
API_ENV=production

# Logging
LOG_LEVEL=INFO

# RAG
FAISS_INDEX_PATH=/app/data/faiss_index
DOCS_PATH=/app/data/docs
```

## Scaling

### Increase Replicas

For load balancing with multiple API instances:

```yaml
version: "3.8"

services:
  api:
    # ... existing config ...
    deploy:
      replicas: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api
```

### Use Environment Variables

```bash
export GROQ_API_KEY=your_key
export ALLOWED_HOSTS=localhost,example.com
docker-compose up -d
```

## Production Configuration

### Use PostgreSQL (Already in compose)

The docker-compose.yml uses PostgreSQL for production-grade database:

```yaml
postgres:
  image: postgres:16-alpine
  environment:
    POSTGRES_USER: beastlife
    POSTGRES_PASSWORD: beastlife_secure_pass
    POSTGRES_DB: beastlife_care
```

### Security Best Practices

1. **Never commit secrets**: Use `.env` file (in `.gitignore`)
2. **Use strong passwords**: Change default PostgreSQL password
3. **Enable SSL**: Use HTTPS in reverse proxy
4. **Restrict networks**: Use private networks in compose

### Override Passwords

```bash
# Create .env.secure
cat > .env.secure << EOF
POSTGRES_PASSWORD=very_strong_password_here
GROQ_API_KEY=xxx
EOF

# Use in compose
docker-compose --env-file .env.secure up -d
```

## Monitoring

### View Real-Time Logs

```bash
# API logs
docker-compose logs -f api

# Dashboard logs
docker-compose logs -f dashboard

# Database logs
docker-compose logs -f postgres
```

### Check Resource Usage

```bash
docker stats
```

### Access Database

```bash
# Open PostgreSQL shell
docker-compose exec postgres psql -U beastlife -d beastlife_care

# Run SQL
SELECT COUNT(*) FROM query_logs;
```

### Monitor API

```bash
# Health check
curl http://localhost:8000/health

# API metrics
curl http://localhost:8000/stats
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs api

# Rebuild image
docker-compose build --no-cache api

# Remove and restart
docker-compose down && docker-compose up -d
```

### Database Connection Error

```bash
# Check if postgres is running
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

### Port Already in Use

```bash
# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Change from 8000 to 8001

# Or find and kill process
# Windows: netstat -ano | findstr :8000
# Linux: lsof -i :8000
```

### API Can't Connect to Database

```bash
# Verify database is healthy
docker-compose ps postgres

# Check network connectivity
docker-compose exec api ping postgres

# Recreate network
docker-compose down
docker network prune
docker-compose up -d
```

## Cleanup

### Remove Containers

```bash
docker-compose down
```

### Remove Images

```bash
docker-compose down --rmi all
```

### Remove All Data (Hard Reset)

```bash
docker-compose down -v
```

### Remove Unused Resources

```bash
docker system prune -a
```

## Deployment to Cloud

### AWS ECS

1. Push image to ECR
2. Create task definition
3. Launch service

### Google Cloud Run

```bash
gcloud run deploy beastlife-api \
  --image gcr.io/project/beastlife-api:latest \
  --platform managed \
  --region us-central1
```

### Heroku

```bash
# Build and push
heroku container:push web
heroku container:release web

# Set environment
heroku config:set GROQ_API_KEY=xxx
```

## Performance Tips

1. **Use PostgreSQL** instead of SQLite for production
2. **Enable Redis caching** for frequently asked questions
3. **Scale API horizontally** behind load balancer
4. **Use GPU support** for faster embeddings
5. **Monitor and optimize** database queries

## Backup & Recovery

### Backup Database

```bash
docker-compose exec postgres pg_dump -U beastlife beastlife_care > backup.sql
```

### Restore Database

```bash
docker-compose exec -T postgres psql -U beastlife beastlife_care < backup.sql
```

### Backup Data

```bash
docker cp beastlife_postgres:/var/lib/postgresql/data ./backup/postgres_data
docker cp beastlife_api:/app/data ./backup/app_data
```

---

For more information, see README.md and QUICKSTART.md
