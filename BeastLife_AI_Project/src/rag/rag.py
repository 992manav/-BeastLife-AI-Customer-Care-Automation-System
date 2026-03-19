import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from src.core.config import get_settings
from src.core.logger import setup_logger
from src.core.models import Document, RetrievalResult

logger = setup_logger(__name__)

class RAGSystem:
    """Retrieval Augmented Generation system using FAISS and SentenceTransformers."""
    
    def __init__(self):
        settings = get_settings()
        self.docs_path = Path(settings.docs_path)
        self.index_path = Path(settings.faiss_index_path)
        self.embedding_model_name = "all-MiniLM-L6-v2"
        
        self.embedding_model = None
        self.index = None
        self.documents: Dict[int, Document] = {}
        self._initialized = False
    
    async def init(self):
        """Initialize RAG system."""
        try:
            logger.info("Initializing RAG system...")
            
            loop = asyncio.get_event_loop()
            self.embedding_model = await loop.run_in_executor(
                None,
                SentenceTransformer,
                self.embedding_model_name
            )
            
            self.docs_path.mkdir(parents=True, exist_ok=True)
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            # Try to load existing index
            if self._index_exists():
                await self._load_index()
            else:
                # Create sample documents and index
                await self._create_sample_documents()
                await self._build_index()
            
            self._initialized = True
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def _index_exists(self) -> bool:
        """Check if index files exist."""
        return (self.index_path / "index.faiss").exists()
    
    async def _load_index(self):
        """Load existing FAISS index."""
        try:
            loop = asyncio.get_event_loop()
            
            self.index = await loop.run_in_executor(
                None,
                lambda: faiss.read_index(str(self.index_path / "index.faiss"))
            )
            
            # Load documents mapping
            docs_file = self.index_path / "documents.json"
            if docs_file.exists():
                with open(docs_file, 'r') as f:
                    docs_data = json.load(f)
                    for idx, doc_dict in docs_data.items():
                        self.documents[int(idx)] = Document(**doc_dict)
            
            logger.info(f"Loaded FAISS index with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise
    
    async def _create_sample_documents(self):
        """Create sample BeastLife documentation."""
        sample_docs = [
            {
                "id": "doc_1",
                "content": "BeastLife is a fitness and wellness platform offering personalized workout plans, nutrition guidance, and community support. Our mission is to help you achieve your fitness goals.",
                "metadata": {"category": "about", "source": "company_info"}
            },
            {
                "id": "doc_2",
                "content": "Workout plans on BeastLife include customized strength training, cardio routines, flexibility programs, and sport-specific training. Plans are tailored to your fitness level and goals.",
                "metadata": {"category": "workouts", "source": "help"}
            },
            {
                "id": "doc_3",
                "content": "Nutrition guidance includes meal plans, macro calculations, supplement recommendations, and hydration tracking. Our experts provide personalized nutrition advice.",
                "metadata": {"category": "nutrition", "source": "help"}
            },
            {
                "id": "doc_4",
                "content": "Membership options: Basic ($9.99/month), Premium ($19.99/month), Elite ($29.99/month). Premium includes 1-on-1 coaching, and Elite includes access to exclusive content.",
                "metadata": {"category": "pricing", "source": "help"}
            },
            {
                "id": "doc_5",
                "content": "Payment methods accepted: Credit cards (Visa, Mastercard, Amex), PayPal, Apple Pay, and Google Pay. If payment fails, check your card details or contact support.",
                "metadata": {"category": "billing", "source": "faq"}
            },
            {
                "id": "doc_6",
                "content": "Refund policy: Full refunds available within 30 days of purchase. After 30 days, we cannot process refunds. Contact support at manavjob992@gmail.com for assistance.",
                "metadata": {"category": "refunds", "source": "policy"}
            },
            {
                "id": "doc_7",
                "content": "Technical issues: If you cannot log in, try resetting your password. For app crashes or slow performance, try clearing cache. If problems persist, contact support.",
                "metadata": {"category": "support", "source": "troubleshooting"}
            },
            {
                "id": "doc_8",
                "content": "Customer support team available Monday-Friday 9AM-6PM EST. Email: manavjob992@gmail.com, Phone: 1-800-BEASTLIFE, Chat: Available on website.",
                "metadata": {"category": "contact", "source": "support"}
            },
        ]
        
        # Save to files
        for doc in sample_docs:
            doc_file = self.docs_path / f"{doc['id']}.json"
            with open(doc_file, 'w') as f:
                json.dump(doc, f)
        
        # Create Document objects
        for i, doc in enumerate(sample_docs):
            self.documents[i] = Document(
                id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"]
            )
        
        logger.info(f"Created {len(sample_docs)} sample documents")
    
    async def _build_index(self):
        """Build FAISS index from documents."""
        try:
            loop = asyncio.get_event_loop()
            
            # Get embeddings for all documents
            contents = [doc.content for doc in self.documents.values()]
            
            embeddings = await loop.run_in_executor(
                None,
                self.embedding_model.encode,
                contents
            )
            
            embeddings = np.array(embeddings).astype('float32')
            
            # Create FAISS index
            def create_index():
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                return index
            
            self.index = await loop.run_in_executor(None, create_index)
            
            # Save index and documents
            await loop.run_in_executor(
                None,
                lambda: faiss.write_index(
                    self.index,
                    str(self.index_path / "index.faiss")
                )
            )
            
            # Save documents mapping
            docs_data = {
                str(idx): {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                }
                for idx, doc in self.documents.items()
            }
            
            with open(self.index_path / "documents.json", 'w') as f:
                json.dump(docs_data, f, indent=2)
            
            logger.info(f"Built and saved FAISS index with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> RetrievalResult:
        """Retrieve relevant documents for a query."""
        try:
            if not self._initialized:
                raise RuntimeError("RAG system not initialized")
            
            loop = asyncio.get_event_loop()
            
            # Get query embedding
            query_embedding = await loop.run_in_executor(
                None,
                self.embedding_model.encode,
                query
            )
            
            query_embedding = np.array([query_embedding]).astype('float32')
            
            # Search index
            def search():
                distances, indices = self.index.search(query_embedding, top_k)
                return distances[0], indices[0]
            
            distances, indices = await loop.run_in_executor(None, search)
            
            # Retrieve documents
            retrieved_docs = []
            scores = []
            
            for distance, idx in zip(distances, indices):
                if idx in self.documents:
                    retrieved_docs.append(self.documents[idx])
                    # Convert distance to similarity score (0-1)
                    similarity = 1 / (1 + distance)
                    scores.append(float(similarity))
            
            return RetrievalResult(
                documents=retrieved_docs,
                scores=scores,
                query=query
            )
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return RetrievalResult(documents=[], scores=[], query=query)
    
    async def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Document],
        llm_provider
    ) -> str:
        """Generate answer from retrieved documents using LLM."""
        try:
            # Build context from documents
            context = "\n\n".join([
                f"Document: {doc.id}\nContent: {doc.content}"
                for doc in retrieved_docs
            ])
            
            prompt = f"""You are BeastLife customer support AI.

Use only the provided documents to answer the user question.

Rules:
- Do not invent facts outside the documents.
- If information is missing, say: "I don't have information about this topic."
- Keep the response concise (2-4 sentences).
- Use a helpful, empathetic customer-support tone.
- If relevant, include one practical next step.

Documents:
{context}

User Question:
{query}

Final Answer:"""
            
            answer = await llm_provider.generate(prompt)
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Sorry, I encountered an error while processing your query."


# Global RAG system instance
rag_system = RAGSystem()

async def get_rag_system() -> RAGSystem:
    """Get RAG system instance."""
    if not rag_system._initialized:
        await rag_system.init()
    return rag_system
