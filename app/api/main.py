# app/api/chat.py
import uuid
import json
import logging
from fastapi import APIRouter, Query, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from bson import ObjectId
from langchain.chat_models import ChatOpenAI

from app.llm.conversation import create_langchain_session, get_chat_history
from app.db.get_db import get_all_products
from app.utils.similarly import find_similar_products_manual
from app.llm.prompts import PRODUCT_QUERY_SCHEMA, QUERY_RELEVANCE_SCHEMA
from app.config import OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

# Lấy danh sách sản phẩm từ DB.
all_products = get_all_products()

# Dictionary lưu trữ session của người dùng.
user_sessions: dict[str, dict] = {}

class UserQuery(BaseModel):
    query: str

@router.get("/init-session")
async def init_session():
    """Tạo session mới và trả về session_id."""
    try:
        session_id = str(uuid.uuid4())
        logger.info(f"Created new session_id: {session_id}")
        user_sessions[session_id] = {
            "conversation": create_langchain_session(),
            "current_product": None
        }
        return {
            "session_id": session_id,
            "detail": "Session initialized",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        return {
            "session_id": None,
            "detail": "Failed to initialize session",
            "status": "error"
        }

def is_query_related_to_previous(history: str, query: str) -> bool:
    """Kiểm tra truy vấn có liên quan đến hội thoại trước không."""
    logger.info("Checking if query is related to the previous conversation...")
    llm = ChatOpenAI(
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4o-mini"
    )

    input_prompt = f"""
    Given the previous conversation history and the current query, determine if the query is related.

    Previous conversation summary:
    {history}

    Current query:
    {query}

    Respond strictly in the following JSON format:
    {json.dumps(QUERY_RELEVANCE_SCHEMA, indent=2)}
    """

    response_text = llm.predict(text=input_prompt)
    
    try:
        response_json = json.loads(response_text)
        return response_json.get("is_related", False)
    except json.JSONDecodeError:
        logger.error("LLM did not return valid JSON format for query relevance check.")
        return False

def is_query_related_to_product(product: dict, query: str) -> bool:
    """Kiểm tra truy vấn có liên quan đến sản phẩm đã lưu không."""
    if not product:
        return False

    product_name = product.get("name", "").lower()
    return product_name in query.lower()

def convert_objectid_to_str(product):
    """Convert ObjectId to string in the product dictionary."""
    if "_id" in product and isinstance(product["_id"], ObjectId):
        product["_id"] = str(product["_id"])
    return product

@router.post("/chat")
async def chat_endpoint(user_query: UserQuery, session_id: str = Query(...)):
    """Process chat query with standardized JSON Schema."""
    logger.info(f"[/chat] Processing query for session {session_id}")
    logger.info(f"User query: {user_query.query}")
    
    try:
        if session_id not in user_sessions:
            logger.error(f"Session {session_id} not found in active sessions")
            return JSONResponse(
                status_code=400,
                content={
                    "session_id": session_id,
                    "response": {
                        "status": "error",
                        "message": "Invalid session. Please refresh the page."
                    }
                }
            )

        session_data = user_sessions[session_id]
        conversation = session_data["conversation"]
        
        # Log the conversation object to debug
        logger.info(f"Conversation object: {conversation}")

        if not user_query.query.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "session_id": session_id,
                    "response": {"status": "error", "message": "Query cannot be empty"}
                }
            )

        # Get chat history
        chat_history = get_chat_history(conversation["history"])
        logger.info(f"Chat history: {chat_history}")

        # Check if the query is related to the previous conversation
        if is_query_related_to_previous(chat_history, user_query.query):
            logger.info("Query is related to the previous conversation.")
        else:
            logger.info("Query is not related to the previous conversation.")

        # Check if the query is related to the current product
        current_product = session_data["current_product"]
        if is_query_related_to_product(current_product, user_query.query):
            logger.info("Query is related to the current product.")
            product_recommendations_text = f"- {current_product['name']}"
        else:
            logger.info("Query is not related to the current product.")
            current_product = None

            # Find similar products
            product_ids, similar_products = find_similar_products_manual(user_query.query, all_products)
            logger.info(f"Found similar products: {similar_products}")

            # Convert ObjectId to string in similar products
            similar_products = [convert_objectid_to_str(product) for product in similar_products]

            # Prepare the product recommendations text
            product_recommendations_text = "\n".join([f"- {product['name']}" for product in similar_products])

        # Prepare the prompt for the LLM
        prompt_text = f"""
        User query: {user_query.query}
        Product recommendations:
        {product_recommendations_text}
        """

        # Send the prompt to the LLM
        response = await conversation["runnable"].ainvoke(
            {"input": prompt_text},
            config={"configurable": {"session_id": session_id}}
        )

        logger.info(f"Response from LLM: {response}")

        return {
            "session_id": session_id,
            "response": {
                "status": "success",
                "answer": str(response.content) if hasattr(response, 'content') else str(response),
                "product_recommendations": similar_products if not current_product else [current_product]
            }
        }

    except Exception as e:
        logger.exception(f"Error in chat_endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "session_id": session_id,
                "response": {
                    "status": "error",
                    "message": f"Internal error: {str(e)}"
                }
            }
        )

app.include_router(router, prefix="")

