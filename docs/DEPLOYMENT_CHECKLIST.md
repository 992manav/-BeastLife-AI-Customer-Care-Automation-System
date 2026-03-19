# Production Deployment Checklist

## Pre-Deployment

### Code Readiness

- [ ] All tests pass: `pytest`
- [ ] No hardcoded credentials in code
- [ ] Error handling implemented for all LLM calls
- [ ] Rate limiting configured
- [ ] Logging configured properly

### Environment Setup

- [ ] Production `.env` file configured
- [ ] API keys obtained and tested
- [ ] Database credentials set with strong passwords
- [ ] SSL/TLS certificates obtained
- [ ] Domain name registered and DNS configured

### Infrastructure

- [ ] PostgreSQL database set up
- [ ] Database backups configured
- [ ] Redis cache (optional) configured
- [ ] Load balancer configured
- [ ] Monitoring tools set up (Prometheus, Grafana)

### Security

- [ ] API authentication implemented
- [ ] Rate limiting enabled
- [ ] CORS properly configured
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention verified
- [ ] XSS prevention measures in place

## Deployment Steps

### 1. Database Preparation

```bash
# Create production database
createdb beastlife_care_prod

# Create user with restricted permissions
createuser beastlife_prod
psql -c "ALTER USER beastlife_prod WITH PASSWORD 'strong_password'"

# Grant permissions
psql beastlife_care_prod -c "GRANT ALL ON SCHEMA public TO beastlife_prod"
```

### 2. Docker Build and Push

```bash
# Build production image
docker build -t beastlife-api:1.0.0 .

# Tag for registry
docker tag beastlife-api:1.0.0 registry.example.com/beastlife-api:1.0.0

# Push to registry
docker push registry.example.com/beastlife-api:1.0.0
```

### 3. Application Deployment

```bash
# Using Docker Compose
docker-compose -f docker-compose.yml up -d

# Or using Kubernetes
kubectl apply -f k8s/
```

### 4. Health Checks

```bash
# Check API health
curl -f https://api.example.com/health || exit 1

# Check database connection
curl -f https://api.example.com/stats || exit 1

# Check dashboard
curl -f https://dashboard.example.com || exit 1
```

## Post-Deployment

### Monitoring Setup

- [ ] Set up Prometheus scraping
- [ ] Configure Grafana dashboards
- [ ] Set up alert rules
- [ ] Configure log aggregation (ELK, CloudWatch)
- [ ] Monitor API response times
- [ ] Monitor error rates

### Testing

- [ ] Smoke tests pass
- [ ] Load testing completed
- [ ] Security scanning completed
- [ ] Penetration testing (if required)

### Documentation

- [ ] Runbook created
- [ ] Incident response plan documented
- [ ] Troubleshooting guide updated
- [ ] Monitoring dashboard screenshots taken

### Backups

- [ ] Database backup configured
- [ ] Backup restoration tested
- [ ] Backup retention policy set (30 days minimum)
- [ ] Off-site backup copy created

### Performance

- [ ] API response time < 500ms (p95)
- [ ] Error rate < 0.1%
- [ ] Database connection pool sized appropriately
- [ ] Cache hit rate > 80%

## Operational Procedures

### Daily

- [ ] Check system health dashboard
- [ ] Review error logs
- [ ] Monitor API metrics
- [ ] Verify backup completion

### Weekly

- [ ] Review performance metrics
- [ ] Check system resource utilization
- [ ] Update security patches
- [ ] Review database growth

### Monthly

- [ ] Review and rotate API keys
- [ ] Analyze query patterns
- [ ] Optimize slow queries
- [ ] Update documentation

### Quarterly

- [ ] Security audit
- [ ] Disaster recovery drill
- [ ] Performance tuning review
- [ ] Capacity planning

## Scaling

### Horizontal Scaling

```docker-compose
api:
  deploy:
    replicas: 3
```

### Vertical Scaling

```env
DATABASE_POOL_SIZE=50
WORKER_THREADS=8
```

### Load Balancer Configuration

```nginx
upstream beastlife {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}
```

## Rollback Plan

### If Issues Detected (< 5% error rate)

1. Stop new deployments
2. Scale up previous stable version
3. Route traffic back to stable version
4. Investigate root cause
5. Fix and redeploy with additional testing

```bash
# Rollback to previous version
docker-compose down
git checkout previous-version
docker build -t beastlife-api:rollback .
docker-compose up -d
```

### Data Restoration

```bash
# Restore from backup
pg_restore -d beastlife_care backup.sql

# Test restored system
curl http://localhost:8000/health
```

## Incident Response

### High CPU Usage

1. Check processes: `docker stats`
2. Review LLM queries: `tail -f logs/app.log`
3. Scale up resources if needed
4. Investigate query patterns

### Database Connection Issues

1. Check connection count: `psql -c "SELECT * FROM pg_stat_activity;"`
2. Increase pool size if needed
3. Monitor for stuck connections
4. Clear connections if necessary

### API Response Delays

1. Check LLM API status
2. Verify database performance
3. Check network latency
4. Review rate limiting

### Memory Leaks

1. Monitor memory over time: `docker stats`
2. Review recent code changes
3. Enable memory profiling
4. Perform garbage collection tuning

## Performance Optimization

### LLM Optimization

- Use faster model for simple queries
- Implement caching for common questions
- Batch similar queries
- Monitor token usage

### Database Optimization

```sql
-- Add indexes
CREATE INDEX idx_customer_id ON query_logs(customer_id);
CREATE INDEX idx_category ON query_logs(category);
CREATE INDEX idx_timestamp ON query_logs(timestamp);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM query_logs WHERE category='order_tracking';
```

### Cache Strategy

```python
# Cache frequently accessed documents
RAG_CACHE_TTL=3600  # 1 hour
QUERY_CACHE_TTL=300  # 5 minutes
```

## Security Checklist

- [ ] All API endpoints require authentication
- [ ] Rate limiting: 100 requests/minute per IP
- [ ] SQL injection prevention verified
- [ ] XSS prevention in place
- [ ] CSRF protection enabled
- [ ] API versioning implemented
- [ ] Secrets not in logs
- [ ] HTTPS only (no HTTP)
- [ ] Security headers configured
- [ ] CORS properly restricted

## Post-Incident Review

After any incident:

1. Create incident report
2. Root cause analysis
3. Implement preventive measures
4. Update runbooks
5. Team debrief
6. Document lessons learned

---

## Monitoring Metrics

### Key Performance Indicators (KPIs)

```
API Metrics:
- Average response time
- P50, P95, P99 latency
- Throughput (requests/sec)
- Error rate (%)
- Availability (%)

Business Metrics:
- Resolution rate (%)
- Escalation rate (%)
- Customer satisfaction
- Cost per query

System Metrics:
- CPU utilization
- Memory utilization
- Database connections
- Cache hit rate
- Token usage (Groq/Gemini)
```

### Alert Thresholds

```
CRITICAL:
- Error rate > 5%
- API down (health check fails)
- Database unavailable
- Response time > 5s

WARNING:
- Error rate > 1%
- Response time > 1s (p95)
- Database connections > 80%
- Memory utilization > 85%
- Cache hit rate < 50%
```

---

**Last Updated:** 2024
**Deployment Frequency:** Weekly
**Rollback Time:** < 15 minutes
**Recovery Time Objective (RTO):** 1 hour
**Recovery Point Objective (RPO):** 15 minutes
