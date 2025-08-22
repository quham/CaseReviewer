#!/usr/bin/env python3
"""
CaseReviewer Python Server
FastAPI-based server for social worker case review searches with PostgreSQL integration
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import numpy as np
from openai import OpenAI
import jwt
import bcrypt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="CaseReviewer API",
    description="Social Worker Case Review Search API with AI-powered insights",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET", "nspcc-case-review-secret")

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

# Initialize DeepInfra API client
try:
    deepinfra_token = os.getenv("DEEPINFRA_TOKEN")
    if not deepinfra_token:
        logger.warning("⚠️ DEEPINFRA_TOKEN not found, embedding functionality will be limited")
        deepinfra_client = None
    else:
        deepinfra_client = OpenAI(
            api_key=deepinfra_token,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        logger.info("✅ DeepInfra API client initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing DeepInfra API client: {e}")
    deepinfra_client = None

# Pydantic models
class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    username: str
    password: str
    name: str
    role: str = "social_worker"
    organization: Optional[str] = None

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")

class SearchResult(BaseModel):
    id: str
    title: str
    summary: str
    child_age: Optional[int]
    risk_types: List[str]
    outcome: Optional[str]
    review_date: Optional[datetime]
    agencies: List[str]
    warning_signs_early: List[str]
    risk_factors: List[str]
    barriers: List[str]
    relationship_model: Dict[str, Any]
    source_file: Optional[str]
    similarity_score: float
    timeline_events: List[Dict[str, Any]]

class CaseReviewDetail(BaseModel):
    id: str
    title: str
    summary: str
    child_age: Optional[int]
    risk_types: List[str]
    outcome: Optional[str]
    review_date: Optional[datetime]
    agencies: List[str]
    warning_signs_early: List[str]
    risk_factors: List[str]
    barriers: List[str]
    relationship_model: Dict[str, Any]
    source_file: Optional[str]
    timeline_events: List[Dict[str, Any]]
    recommendations: List[str]
    key_lessons: List[str]

# Database connection class
class DatabaseManager:
    def __init__(self):
        self.database_url = DATABASE_URL
        
    def get_connection(self):
        return psycopg2.connect(self.database_url)
    
    async def execute_query(self, query: str, params: tuple = None, fetch: bool = True):
        """Execute a database query with proper connection management"""
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                else:
                    conn.commit()
                    return cursor.rowcount
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

# Initialize database manager
db_manager = DatabaseManager()

# Authentication functions
def create_jwt_token(user_data: dict) -> str:
    """Create JWT token for user"""
    payload = {
        "id": user_data["id"],
        "username": user_data["username"],
        "role": user_data["role"],
        "exp": datetime.utcnow().timestamp() + (24 * 60 * 60)  # 24 hours
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_jwt_token(token: str) -> dict:
    """Verify JWT token and return user data"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Get current authenticated user"""
    token = credentials.credentials
    user_data = verify_jwt_token(token)
    return user_data

# AI-powered functions
def create_embedding(text: str) -> List[float]:

    
    try:
        # Truncate text if too long
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars]
        
        # Use DeepInfra API for embeddings
        result = deepinfra_client.embeddings.create(
            model="Qwen/Qwen3-Embedding-8B",
            input=text,
            encoding_format="float"
        )
        
        # Convert to list of floats
        embedding = result.data[0].embedding
        return embedding
    except Exception as e:
        logger.error(f"Error creating embedding via DeepInfra API: {e}")
    

# Import OpenRouter insights service
try:
    from services.openrouter_insights import openrouter_insights_service
    logger.info("✅ OpenRouter insights service loaded")
except ImportError as e:
    logger.warning(f"⚠️ OpenRouter insights service not available: {e}")
    openrouter_insights_service = None



def extract_recommendations_and_lessons(case_data: dict) -> tuple[List[str], List[str]]:
    """Extract recommendations and key lessons from case data"""
    recommendations = []
    key_lessons = []
    
    # Extract from barriers (what went wrong)
    if case_data.get("barriers"):
        for barrier in case_data["barriers"]:
            key_lessons.append(f"Barrier: {barrier}")
            recommendations.append(f"Address: {barrier}")
    
    # Extract from warning signs
    if case_data.get("warning_signs_early"):
        for sign in case_data["warning_signs_early"]:
            key_lessons.append(f"Early warning: {sign}")
            recommendations.append(f"Monitor for: {sign}")
    
    # Extract from risk factors
    if case_data.get("risk_factors"):
        for factor in case_data["risk_factors"]:
            key_lessons.append(f"Risk factor: {factor}")
            recommendations.append(f"Mitigate: {factor}")
    
    # Extract from outcome
    if case_data.get("outcome"):
        key_lessons.append(f"Outcome: {case_data['outcome']}")
    
    return recommendations[:5], key_lessons[:5]

# API Routes
@app.post("/api/register")
async def register(user_data: UserRegister):
    """Register a new user"""
    try:
        # Check if username already exists
        existing_user = await db_manager.execute_query(
            "SELECT id FROM users WHERE username = %s",
            (user_data.username,)
        )
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Hash password
        hashed_password = bcrypt.hashpw(user_data.password.encode(), bcrypt.gensalt()).decode()
        
        # Insert user
        result = await db_manager.execute_query(
            """
            INSERT INTO users (username, password, name, role, organization)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, username, name, role
            """,
            (user_data.username, hashed_password, user_data.name, user_data.role, user_data.organization),
            fetch=False
        )
        
        # Get the created user
        user = await db_manager.execute_query(
            "SELECT id, username, name, role FROM users WHERE username = %s",
            (user_data.username,)
        )
        
        if user:
            user_data_dict = dict(user[0])
            token = create_jwt_token(user_data_dict)
            return {
                "token": token,
                "user": user_data_dict
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create user")
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/login")
async def login(user_credentials: UserLogin):
    """Login user"""
    try:
        # Get user by username
        user = await db_manager.execute_query(
            "SELECT id, username, password, name, role FROM users WHERE username = %s",
            (user_credentials.username,)
        )
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user_data = dict(user[0])
        
        # Verify password - handle both string and bytes
        stored_password = user_data["password"]
        if isinstance(stored_password, str):
            stored_password = stored_password.encode('utf-8')
        
        if not bcrypt.checkpw(user_credentials.password.encode('utf-8'), stored_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create token
        token = create_jwt_token(user_data)
        
        return {
            "token": token,
            "user": {
                "id": user_data["id"],
                "username": user_data["username"],
                "name": user_data["name"],
                "role": user_data["role"]
            }
        }
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/protected/search")
async def search_case_reviews(
    search_query: SearchQuery,
    current_user: dict = Depends(get_current_user)
):
    """Search case reviews with AI-powered insights"""
    try:
        # Check if DeepInfra client is working
        if not deepinfra_client:
            logger.warning("⚠️ DeepInfra client not available - using OpenRouter fallback")
            
            # Generate general advice using OpenRouter without case matching
            general_advice = None
            if openrouter_insights_service:
                try:
                    # Create a simple prompt for general advice
                    general_prompt = f"""
You are an expert social work professional with extensive experience in child protection and social work practice.

A social worker has searched for guidance on: "{search_query.query}"

Provide one or two clear, practical paragraphs of professional advice  that addresses their query. Consider:

- Professional standards and social work best practices
- Evidence-based approaches
- Practical steps they can take
- When to seek supervision or escalate

Keep your response concise but comprehensive. Focus on actionable guidance that any social worker can apply immediately.
"""
                    
                    # Get general advice from OpenRouter
                    general_result = openrouter_insights_service._call_openrouter(general_prompt, max_tokens=300)
                    
                    if general_result:
                        general_advice, used_model = general_result
                        # Clean up the response
                        general_advice = general_advice.strip()
                        # Remove any markdown formatting if present
                        if general_advice.startswith('```'):
                            general_advice = general_advice.split('```')[1] if len(general_advice.split('```')) > 1 else general_advice
                        if general_advice.startswith('**') or general_advice.startswith('*'):
                            general_advice = general_advice.replace('**', '').replace('*', '')
                    
                    logger.info(f"✅ Generated general advice using OpenRouter model: {used_model}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate general advice: {e}")
                    general_advice = None
            
            # Return response with general advice instead of case matches
            response_data = {
                "results": [],
                "totalCount": 0,
                "searchTime": 0,
                "message": general_advice if general_advice else "Unable to provide specific guidance at this time. Please consult your supervisor or refer to your organization's policies and procedures.",
                "search_status": "fallback_mode"
            }
            
            # Save search to history
            await db_manager.execute_query(
                """
                INSERT INTO searches (user_id, query, filters, results_count, searched_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (current_user["id"], search_query.query, Json(search_query.filters or {}), 0, datetime.now()),
                fetch=False
            )
            
            return response_data
        
        # Original embedding-based search logic continues here...
        # Create embedding for search query
        query_embedding = create_embedding(search_query.query)
        
        # Vector similarity search
        results = await db_manager.execute_query(
            """
            SELECT 
                id, title, summary, child_age, risk_types, outcome, 
                review_date, agencies, warning_signs_early, risk_factors, 
                barriers, relationship_model, source_file, created_at,
                1 - (embedding <=> %s::vector) as similarity_score
            FROM case_reviews
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, search_query.top_k)
        )
        
        if not results:
            return {
                "results": [],
                "totalCount": 0,
                "searchTime": 0,
                "message": "No relevant cases found"
            }
        
        # Process results with personalized insights
        processed_results = []
        for result in results:
            case_data = dict(result)
            
            # Get timeline events
            timeline_events = await db_manager.execute_query(
                "SELECT * FROM timeline_events WHERE case_review_id = %s ORDER BY event_date",
                (case_data["id"],)
            )
            
            # Debug logging
            logger.info(f"Case {case_data['id']}: Found {len(timeline_events)} timeline events")
            if timeline_events:
                logger.info(f"Timeline events for case {case_data['id']}: {[dict(event) for event in timeline_events]}")
                # Log the first event structure
                first_event = dict(timeline_events[0]) if timeline_events else None
                if first_event:
                    logger.info(f"First event structure: {first_event}")
                    logger.info(f"First event keys: {list(first_event.keys())}")
            else:
                logger.warning(f"No timeline events found for case {case_data['id']}")
            
            # Create search result
            search_result = SearchResult(
                id=str(case_data["id"]),
                title=case_data["title"],
                summary=case_data["summary"],
                child_age=case_data["child_age"],
                risk_types=case_data["risk_types"] or [],
                outcome=case_data["outcome"],
                review_date=case_data["review_date"],
                agencies=case_data["agencies"] or [],
                warning_signs_early=case_data["warning_signs_early"] or [],
                risk_factors=case_data["risk_factors"] or [],
                barriers=case_data["barriers"] or [],
                relationship_model=case_data["relationship_model"] or {},
                source_file=case_data["source_file"],
                similarity_score=float(case_data["similarity_score"]),
                timeline_events=[dict(event) for event in timeline_events]
            )
            
            processed_results.append(search_result)
        
        # Generate comprehensive AI-powered recommendations using OpenRouter
        comprehensive_recommendations = None
        if openrouter_insights_service:
            try:
                comprehensive_recommendations = openrouter_insights_service.generate_personalized_recommendations(
                    search_query.query,
                    [result.dict() for result in processed_results[:5]],  # Convert SearchResult to dicts
                    current_user.get("role", "social_worker")
                )
                logger.info("✅ Generated comprehensive AI recommendations")
            except Exception as e:
                logger.error(f"Failed to generate comprehensive recommendations: {e}")
                comprehensive_recommendations = None
        
        # Save search to history
        await db_manager.execute_query(
            """
            INSERT INTO searches (user_id, query, filters, results_count, searched_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (current_user["id"], search_query.query, Json(search_query.filters or {}), len(processed_results), datetime.now()),
            fetch=False
        )
        
        response_data = {
            "results": [result.dict() for result in processed_results],
            "totalCount": len(processed_results),
            "searchTime": 0,  # Could implement actual timing
            "message": f"Found {len(processed_results)} relevant cases"
        }
        
        # Add comprehensive AI recommendations if available
        if comprehensive_recommendations:
            response_data["ai_recommendations"] = comprehensive_recommendations
            response_data["message"] += " with AI-powered insights"
        
        return response_data
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.get("/api/protected/case-reviews/{case_id}")
async def get_case_review_detail(
    case_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed case review with timeline and recommendations"""
    try:
        # Get case review
        case_review = await db_manager.execute_query(
            "SELECT * FROM case_reviews WHERE id = %s",
            (case_id,)
        )
        
        if not case_review:
            raise HTTPException(status_code=404, detail="Case review not found")
        
        case_data = dict(case_review[0])
        
        # Get timeline events
        timeline_events = await db_manager.execute_query(
            "SELECT * FROM timeline_events WHERE case_review_id = %s ORDER BY event_date",
            (case_id,)
        )
        
        # Debug logging
        logger.info(f"Case review {case_id}: Found {len(timeline_events)} timeline events")
        if timeline_events:
            logger.info(f"Timeline events for case review {case_id}: {[dict(event) for event in timeline_events]}")
            # Log the first event structure
            first_event = dict(timeline_events[0]) if timeline_events else None
            if first_event:
                logger.info(f"First event structure: {first_event}")
                logger.info(f"First event keys: {list(first_event.keys())}")
        else:
            logger.warning(f"No timeline events found for case review {case_id}")
        
        # Extract recommendations and lessons
        recommendations, key_lessons = extract_recommendations_and_lessons(case_data)
        
        # Create detailed case review
        detailed_case = CaseReviewDetail(
            id=case_data["id"],
            title=case_data["title"],
            summary=case_data["summary"],
            child_age=case_data["child_age"],
            risk_types=case_data["risk_types"] or [],
            outcome=case_data["outcome"],
            review_date=case_data["review_date"],
            agencies=case_data["agencies"] or [],
            warning_signs_early=case_data["warning_signs_early"] or [],
            risk_factors=case_data["risk_factors"] or [],
            barriers=case_data["barriers"] or [],
            relationship_model=case_data["relationship_model"] or {},
            source_file=case_data["source_file"],
            timeline_events=[dict(event) for event in timeline_events],
            recommendations=recommendations,
            key_lessons=key_lessons
        )
        
        return detailed_case
        
    except Exception as e:
        logger.error(f"Case review fetch error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch case review")

@app.get("/api/protected/search-history")
async def get_search_history(current_user: dict = Depends(get_current_user)):
    """Get user's search history"""
    try:
        searches = await db_manager.execute_query(
            "SELECT * FROM searches WHERE user_id = %s ORDER BY searched_at DESC",
            (current_user["id"],)
        )
        
        return [dict(search) for search in searches]
        
    except Exception as e:
        logger.error(f"Search history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch search history")

@app.get("/api/protected/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return {"user": current_user}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Serve static files for the frontend
if os.path.exists("../client/dist"):
    app.mount("/", StaticFiles(directory="../client/dist", html=True), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
