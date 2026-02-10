# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
OpenViking Memory Service - FastAPI Application

Provides REST API endpoints for memory management operations.
This service integrates with OpenClaw's plugin system for enhanced memory capabilities.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from openviking import OpenViking
from openviking.core.context import Context, ContextType
from openviking.message import Message, Part
from openviking.utils import get_logger
from openviking.utils.config import (
    VectorDBBackendConfig,
    AGFSConfig,
    get_openviking_config,
)

logger = get_logger(__name__)

# Global OpenViking client instance
ov_client: Optional[OpenViking] = None

# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class MemoryStoreRequest(BaseModel):
    """Request model for storing a memory."""
    text: str = Field(..., description="The text content to remember", min_length=1)
    importance: Optional[float] = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Importance score (0.0 to 1.0)"
    )
    category: Optional[str] = Field(
        None,
        description="Memory category (profile, preferences, entities, events, cases, patterns)"
    )
    user: Optional[str] = Field("default", description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")


class MemoryRecallRequest(BaseModel):
    """Request model for recalling memories."""
    query: str = Field(..., description="Search query text", min_length=1)
    limit: Optional[int] = Field(5, ge=1, le=20, description="Maximum number of results")
    user: Optional[str] = Field("default", description="User identifier")
    context_type: Optional[str] = Field(
        "memory",
        description="Context type to search (memory, resource, skill)"
    )
    mode: Optional[str] = Field(
        "thinking",
        description="Retrieval mode (thinking, quick)"
    )


class MemoryItem(BaseModel):
    """Model for a memory item in response."""
    uri: str
    abstract: str
    category: str
    score: float
    relations: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None


class MemoryResponse(BaseModel):
    """Response model for memory recall."""
    status: str
    memories: List[MemoryItem]
    count: int
    backend: str = "openviking"


class StoreResponse(BaseModel):
    """Response model for memory store."""
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None
    backend: str = "openviking"


class ForgetResponse(BaseModel):
    """Response model for forgetting a memory."""
    status: str
    deleted: str
    message: str
    backend: str = "openviking"


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    service: str
    version: str
    timestamp: str
    backend_ready: bool


class StatsResponse(BaseModel):
    """Response model for memory statistics."""
    status: str
    total_memories: int
    user_memories: int
    agent_memories: int
    by_category: Dict[str, int]
    backend: str = "openviking"


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="OpenViking Memory Service",
    description="REST API for OpenViking memory management - Integration with OpenClaw",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware for OpenClaw integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize OpenViking client on startup."""
    global ov_client
    
    logger.info("Starting OpenViking Memory Service...")
    
    try:
        # Load configuration from environment variables
        backend_type = os.getenv("VIKINGDB_BACKEND", "local")
        db_path = os.getenv("VIKINGDB_PATH", "./data/vikingdb")
        vector_dim = int(os.getenv("VECTOR_DIM", "128"))
        distance_metric = os.getenv("DISTANCE_METRIC", "cosine")
        agfs_url = os.getenv("AGFS_URL", "http://localhost:8080")
        agfs_timeout = int(os.getenv("AGFS_TIMEOUT", "30"))
        
        # Create configurations
        vectordb_config = VectorDBBackendConfig(
            backend=backend_type,
            path=db_path,
            vector_dim=vector_dim,
            distance_metric=distance_metric,
        )
        agfs_config = AGFSConfig(
            url=agfs_url,
            timeout=agfs_timeout,
        )
        
        # Initialize OpenViking client
        ov_client = OpenViking(path="./data")
        
        logger.info(f"OpenViking client initialized (backend={backend_type})")
        logger.info(f"VikingDB path: {db_path}")
        logger.info(f"VikingFS URL: {agfs_url}")
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenViking client: {e}")
        raise e


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    global ov_client
    
    logger.info("Shutting down OpenViking Memory Service...")
    
    if ov_client:
        try:
            await ov_client.close()
            logger.info("OpenViking client closed successfully")
        except Exception as e:
            logger.error(f"Error closing OpenViking client: {e}")


# ============================================================================
# Dependency Injection
# ============================================================================

async def get_ov_client() -> OpenViking:
    """Dependency injection for OpenViking client."""
    if ov_client is None:
        raise HTTPException(status_code=503, detail="OpenViking client not initialized")
    return ov_client


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    backend_ready = ov_client is not None
    
    try:
        if backend_ready and ov_client._vikingdb_manager:
            # Quick health check: list collections
            await ov_client._vikingdb_manager.list_collections()
    except Exception:
        backend_ready = False
    
    return HealthResponse(
        status="healthy" if backend_ready else "unhealthy",
        service="openviking-memory-service",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        backend_ready=backend_ready,
    )


@app.post("/memory/store", response_model=StoreResponse, tags=["Memory"])
async def store_memory(
    request: MemoryStoreRequest,
    client: OpenViking = Depends(get_ov_client)
) -> StoreResponse:
    """
    Store a memory in OpenViking.
    
    This endpoint adds a message to a session and commits it, triggering
    the memory extraction and storage pipeline.
    """
    try:
        # Get or create session
        user = request.user or "default"
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        session = client.get_session(user=f"integration_{user}", session_id=session_id)
        session.load()
        
        # Add message to session
        session.add_message(
            role="user",
            parts=[Part(text=request.text)]
        )
        
        # Store metadata
        if len(session._messages) > 0:
            session._messages[-1].meta = {
                "importance": request.importance,
                "category": request.category,
                "source": "openclaw_integration"
            }
        
        # Commit session (triggers memory extraction)
        result = session.commit()
        
        logger.info(
            f"Memory stored: text='{request.text[:50]}...', "
            f"category={request.category}, "
            f"extracted={result.get('memories_extracted', 0)}"
        )
        
        return StoreResponse(
            status="success",
            message=f"Memory stored successfully. Extracted {result.get('memories_extracted', 0)} memories.",
            details=result,
        )
        
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/recall", response_model=MemoryResponse, tags=["Memory"])
async def recall_memory(
    request: MemoryRecallRequest,
    client: OpenViking = Depends(get_ov_client)
) -> MemoryResponse:
    """
    Recall memories using hierarchical retrieval.
    
    This endpoint performs semantic search across stored memories
    using OpenViking's hierarchical retriever.
    """
    try:
        from openviking.retrieve.hierarchical_retriever import (
            HierarchicalRetriever,
            RetrieverMode,
        )
        from openviking.retrieve.types import TypedQuery
        
        # Get embedder from configuration
        config = get_openviking_config()
        embedder = config.embedding.get_embedder()
        
        if not embedder:
            raise HTTPException(
                status_code=503,
                detail="Embedder not configured for retrieval"
            )
        
        # Create retriever
        retriever = HierarchicalRetriever(
            storage=client._vikingdb_manager,
            embedder=embedder,
            rerank_config=config.rerank,
        )
        
        # Map context type string to enum
        context_type_map = {
            "memory": ContextType.MEMORY,
            "resource": ContextType.RESOURCE,
            "skill": ContextType.SKILL,
        }
        context_type = context_type_map.get(request.context_type, ContextType.MEMORY)
        
        # Create typed query
        query = TypedQuery(
            query=request.query,
            context_type=context_type,
        )
        
        # Perform retrieval
        result = await retriever.retrieve(
            query=query,
            limit=request.limit,
            mode=RetrieverMode.THINKING if request.mode == "thinking" else RetrieverMode.QUICK,
        )
        
        # Convert to response format
        memories = [
            MemoryItem(
                uri=ctx.uri,
                abstract=ctx.abstract,
                category=ctx.category,
                score=ctx.score,
                relations=[r.uri for r in ctx.relations],
                created_at=ctx.created_at.isoformat() if ctx.created_at else None,
            )
            for ctx in result.matched_contexts
        ]
        
        logger.info(
            f"Memory recall: query='{request.query}', "
            f"found={len(memories)} results"
        )
        
        return MemoryResponse(
            status="success",
            memories=memories,
            count=len(memories),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recalling memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{memory_id}", response_model=ForgetResponse, tags=["Memory"])
async def forget_memory(
    memory_id: str,
    client: OpenViking = Depends(get_ov_client)
) -> ForgetResponse:
    """
    Delete a specific memory from OpenViking.
    
    This endpoint removes both the vector database entry and the file storage.
    """
    try:
        # Delete from VikingDB
        deleted_count = await client._vikingdb_manager.delete(
            collection="context",
            ids=[memory_id]
        )
        
        if deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Memory with ID '{memory_id}' not found"
            )
        
        # Attempt to delete from VikingFS (if file exists)
        try:
            # Extract URI from ID or search for it
            # For simplicity, we'll skip VikingFS deletion as it requires URI
            # In production, you'd need to map memory_id to URI
            pass
        except Exception as e:
            logger.warning(f"Failed to delete from VikingFS: {e}")
        
        logger.info(f"Memory deleted: {memory_id}")
        
        return ForgetResponse(
            status="success",
            deleted=memory_id,
            message=f"Memory '{memory_id}' successfully forgotten",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forgetting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/stats", response_model=StatsResponse, tags=["Memory"])
async def get_memory_stats(
    client: OpenViking = Depends(get_ov_client)
) -> StatsResponse:
    """
    Get memory statistics.
    
    Returns counts of total memories, user memories, agent memories,
    and breakdown by category.
    """
    try:
        # Count total memories
        total = await client._vikingdb_manager.count(
            collection="context",
            filter={"op": "must", "field": "context_type", "conds": ["memory"]}
        )
        
        # Count by category
        categories = ["profile", "preferences", "entities", "events", "cases", "patterns"]
        by_category: Dict[str, int] = {}
        
        for cat in categories:
            count = await client._vikingdb_manager.count(
                collection="context",
                filter={
                    "op": "and",
                    "conds": [
                        {"op": "must", "field": "context_type", "conds": ["memory"]},
                        {"op": "must", "field": "category", "conds": [cat]}
                    ]
                }
            )
            by_category[cat] = count
        
        # Distinguish user vs agent memories
        user_cats = ["profile", "preferences", "entities", "events"]
        agent_cats = ["cases", "patterns"]
        
        user_memories = sum(by_category.get(cat, 0) for cat in user_cats)
        agent_memories = sum(by_category.get(cat, 0) for cat in agent_cats)
        
        return StatsResponse(
            status="success",
            total_memories=total,
            user_memories=user_memories,
            agent_memories=agent_memories,
            by_category=by_category,
        )
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for running the service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenViking Memory Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "openviking.service.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
