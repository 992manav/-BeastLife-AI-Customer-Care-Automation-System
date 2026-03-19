# Sample Customer Queries Dataset

This document contains example customer queries across all problem categories supported by the BeastLife AI System.

## 📊 Query Categories

The system classifies customer queries into the following 7 primary categories:

1. **Order Tracking** (35% expected)
2. **Delivery Delays** (22% expected)
3. **Refund Requests** (18% expected)
4. **Product Complaints** (15% expected)
5. **Payment Issues** (5% expected)
6. **Subscription Issues** (3% expected)
7. **General Questions** (2% expected)

---

## 1. Order Tracking Queries (35% of total)

Expected to be resolved via **Path A (API Resolution)**

### Query Set A1

```
"Can you tell me the status of my order #ORD-98765?"
"Where is my package? Order number ORD-45321"
"How do I track my order? My order ID is ORD-11122"
"What's the status of order #ORD-56789? I placed it last week"
"Can you provide tracking information for my recent purchase?"
"My order says it's shipped, but where is it? Order #ORD-77889"
"I need to know when my order will arrive. Order #ORD-33445"
"Has my order been dispatched? Reference: ORD-99887"
"When will my fitness equipment arrive? Order #ORD-22334"
"Can you check the delivery status of order ORD-55443?"
```

---

## 2. Delivery Delays (22% of total)

Expected to be resolved via **Path B (RAG Knowledge Base)** + potential escalation

### Query Set B1

```
"My order was supposed to arrive 3 days ago and hasn't shown up!"
"It's been 2 weeks and I still haven't received my package. This is unacceptable!"
"Why is my delivery taking so long? Expected delivery date has passed"
"My order is delayed. What's happening?"
"I'm frustrated - my package is late by almost a week now"
"Expected delivery was yesterday but my package isn't here. What's going on?"
"How long do I have to wait for my order? It's already delayed"
"This is taking way too long. When will I finally get my order?"
"My delivery is overdue. Can someone help me?"
"Why is there so much delay in shipping my order?"
```

---

## 3. Refund Requests (18% of total)

Expected to be resolved via **Path A (API Resolution)**

### Query Set C1

```
"I want to return this product and get a refund"
"Can I get my money back? I'm not satisfied with my purchase"
"Please refund my payment. I changed my mind about this purchase"
"I'd like to initiate a refund for order #ORD-12345"
"How do I get a refund? I want to return my purchase"
"My product arrived damaged. I need a full refund"
"Can you process a return and refund for me?"
"I'm not happy with this product. Please give me my money back"
"I'd like to cancel my order and get refunded. Order #ORD-67890"
"The item doesn't work as advertised. I want a refund"
```

---

## 4. Product Complaints (15% of total)

Expected to be resolved via **Path B (RAG Knowledge Base)** + potential escalation

### Query Set D1

```
"The product I received is broken!"
"This fitness tracker stopped working after 2 days"
"The product quality is terrible. Not as described in the listing"
"I received a defective item. What do I do?"
"The app keeps crashing on my phone. This is frustrating!"
"The product doesn't match the description. Very disappointed"
"This supplement tastes awful. Is this normal?"
"The equipment arrived with missing parts!"
"The product is not working properly. Help!"
"I'm getting errors with this software. Very dissatisfied"
```

---

## 5. Payment Issues (5% of total)

Expected to be resolved via **Path A (API Resolution)**

### Query Set E1

```
"My payment got declined. Can you help?"
"I've been charged twice for the same order!"
"Why was I charged multiple times? This is a mistake"
"My card payment keeps failing. What's wrong?"
"I see duplicate charges on my credit card"
"My payment method is being rejected. Why?"
"I was double-charged. Please refund the extra amount"
"The payment system isn't accepting my card"
"Why is my payment repeatedly failing?"
"I see unauthorized charges on my account"
```

---

## 6. Subscription Issues (3% of total)

Expected to be resolved via **Path A or B**

### Query Set F1

```
"How do I cancel my subscription?"
"I want to upgrade my membership plan"
"Can I pause my subscription temporarily?"
"I was charged for a subscription I don't want"
"How do I change my subscription tier?"
"I want to downgrade my membership to save money"
"When is my subscription renewal? Can I disable auto-renewal?"
"Can I get a discount on my subscription renewal?"
"Is there a cheaper subscription plan?"
"I want to switch from monthly to annual subscription"
```

---

## 7. General Questions (2% of total)

Expected to be resolved via **Path B (RAG Knowledge Base)**

### Query Set G1

```
"What are your business hours?"
"How can I contact customer support?"
"Do you have a physical store location?"
"What's your return policy?"
"How do I update my account information?"
"Do you offer international shipping?"
"What payment methods do you accept?"
"How do I create an account?"
"Do you have gift cards available?"
"What's the warranty on your products?"
```

---

## 📈 Sentiment Distribution

### High-Confidence, Positive Sentiment (Path A Easily)

```
"I love your product! I just want to track my order #ORD-12345"
"Thanks for the great service! Quick question about my order status"
"Everything is great! Just need help with tracking"
```

### Negative Sentiment (May escalate to Path C)

```
"I'm VERY upset! My order is missing and it's been 2 weeks!"
"UNACCEPTABLE! This is the worst customer service ever!"
"I'm furious! I've been charged THREE times and nobody is helping!"
```

### Critical Sentiment (Path C - Escalation)

```
"THIS IS A SCAM! I've been charged but no product arrived!"
"CRITICAL PROBLEM! My account has been compromised and I'm being charged for things I didn't buy!"
"URGENT: I'm experiencing fraud on my account, please help immediately!"
```

---

## 🔄 Multi-Intent Queries (Advanced)

Queries that may require multiple classifications:

```
"My order #ORD-45678 is delayed by a week AND I want a refund"
→ Status: Delivery Delay + Refund Request

"I received a broken product and I'm also having payment issues with my billing"
→ Status: Product Complaint + Payment Issue

"Can I cancel my subscription and get the refund for last month's charge?"
→ Status: Subscription Issue + Refund Request

"The app crashes when I try to track my order. Please fix this!"
→ Status: General Question + Technical Issue

"My delivery is missing AND I was overcharged on my payment method"
→ Status: Delivery Delay + Payment Issue
```

---

## 📝 Testing Dataset

### Minimum Test Set (7 queries - 1 per category)

1. **Order Tracking**: "Can you track my order #ORD-98765?"
2. **Delivery Delays**: "My order was supposed to arrive 3 days ago - where is it?"
3. **Refund Request**: "I want to return this product and get a refund"
4. **Product Complaint**: "The product I received is broken!"
5. **Payment Issue**: "My payment got declined. Can you help?"
6. **Subscription Issue**: "How do I cancel my subscription?"
7. **General Question**: "What's your return policy?"

### Extended Test Set (21 queries - 3 per category)

Run all Query Sets above (A1, B1, C1, D1, E1, F1, G1) for comprehensive testing.

### Stress Test Set (100+ queries)

Process all queries from all sets to test:

- System throughput and latency
- Category distribution accuracy
- Path routing accuracy
- Sentiment analysis accuracy
- Database storage and query performance

---

## 🚀 Using the Sample Queries

### Option 1: CLI Testing

```bash
python main.py test-query "Can you track my order #ORD-98765?"
```

### Option 2: Batch Test

```bash
python main.py batch-test SAMPLE_QUERIES.md
```

### Option 3: API Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Can you track my order #ORD-98765?"}'
```

### Option 4: Dashboard Upload

1. Start the Streamlit dashboard: `python main.py dashboard`
2. Use the "Upload Query Dataset" feature
3. Select `SAMPLE_QUERIES.md`
4. View real-time categorization and metrics

---

## 📊 Expected Output Format

For each query, the system should return:

```json
{
  "query": "Can you track my order #ORD-98765?",
  "category": "order_tracking",
  "sentiment": "neutral",
  "confidence": 0.92,
  "path": "A",
  "response": "Your order #ORD-98765 is being processed. Expected delivery: 3-5 business days",
  "entities": {
    "order_ids": ["ORD-98765"]
  },
  "execution_time_ms": 245.67
}
```

---

## 📌 Notes

- All order numbers are examples and do not represent actual orders
- Queries are realistic sample data for demonstration purposes
- The actual distribution may vary based on real customer data
- All sentiment levels are included for comprehensive testing
- Mix of structured and unstructured query formats included
