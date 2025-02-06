import os
import uuid
import logging

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain modules
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain  # Lưu ý: Deprecated trong LangChain 0.2.7, dùng tạm cho dự án hiện tại.
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate

# Giả sử các module dưới đây đã được triển khai theo yêu cầu của bạn
from get_db import get_all_products
from similarly import find_similar_products_manual

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lấy tất cả sản phẩm từ DB
all_products = get_all_products()

app = FastAPI()

# Cấu hình CORS (ví dụ: frontend chạy tại http://localhost:5500)
frontend_origin = "http://localhost:5500"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary lưu trữ session của người dùng.
# Mỗi session chứa một conversation chain và thông tin sản phẩm hiện tại (nếu có).
user_sessions: dict[str, dict] = {}

def create_langchain_session() -> ConversationChain:
    """
    Tạo và trả về một conversation chain mới sử dụng summarized memory
    để lưu lại những ý chính của cuộc hội thoại.
    """
    logger.info("Creating new LangChain session with summarized memory...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    llm = ChatOpenAI(
        temperature=0.7,
        openai_api_key=api_key,
        model="gpt-4o-mini"
    )

    # Sử dụng ConversationSummaryBufferMemory để tóm tắt lịch sử hội thoại
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=500)
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""\
Previous conversation:
{history}

Customer: {input}
Assistant:"""
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )
    return conversation

class UserQuery(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    """Kiểm tra biến môi trường cần thiết khi khởi động ứng dụng."""
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set")
        raise ValueError("OPENAI_API_KEY is not set")
    logger.info("Startup checks completed successfully.")

@app.get("/init-session")
async def init_session():
    """
    Tạo session mới và trả về session_id.
    Mỗi session lưu một conversation chain và thông tin sản phẩm hiện tại (ban đầu là None).
    """
    session_id = str(uuid.uuid4())
    logger.info(f"Created new session_id: {session_id}")
    user_sessions[session_id] = {
        "conversation": create_langchain_session(),
        "current_product": None  # Chưa có sản phẩm nào được lưu
    }
    return {
        "session_id": session_id,
        "detail": "Session initialized"
    }

def is_query_related_to_previous(history: str, query: str) -> bool:
    """
    Dùng LLM với nhiệt độ 0 để xác định truy vấn hiện tại có liên quan đến cuộc hội thoại trước không.
    Trả về True nếu liên quan, ngược lại trả về False.
    """
    logger.info("Checking if query is related to the previous conversation...")
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        temperature=0.0,
        openai_api_key=api_key,
        model="gpt-4o-mini"
    )

    input_prompt = f"""
Previous conversation:
{history}

Current query:
{query}

Is the current query asking about the same product or related to the previous conversation?
Respond with "Yes" or "No" only.
"""
    response = llm.predict(text=input_prompt)
    return response.strip().lower() == "yes"

def is_query_related_to_product(product: dict, query: str) -> bool:
    """
    Kiểm tra xem truy vấn hiện tại có liên quan đến sản phẩm đã lưu trong session không.
    Ở đây dùng cách đơn giản: nếu tên sản phẩm xuất hiện trong truy vấn thì coi như liên quan.
    """
    if product is None:
        return False

    # Giả sử product là dict có trường "name", nếu product là kiểu khác, chuyển đổi sang string
    product_name = product.get("name", "").lower() if isinstance(product, dict) else str(product).lower()
    return product_name in query.lower()

@app.post("/chat/")
async def chat_endpoint(
    user_query: UserQuery,
    session_id: str = Query(None)
):
    """
    Endpoint xử lý truy vấn chat:
    - Nếu không có session_id, tạo session mới.
    - Nếu truy vấn không liên quan đến cuộc hội thoại trước, reset memory và thông tin sản phẩm.
    - Nếu sản phẩm hiện tại chưa được lưu hoặc truy vấn không liên quan đến sản phẩm đã lưu, thực hiện truy vấn DB.
    - Xây dựng prompt cuối cùng kết hợp thông tin truy vấn và sản phẩm, sau đó trả về phản hồi từ LLM.
    """
    logger.info(f"[/chat] Received chat query: {user_query.query}")

    # Tạo session mới nếu không có hoặc session_id không tồn tại
    if not session_id or session_id not in user_sessions:
        session_id = str(uuid.uuid4())
        logger.info(f"Session not provided or not found, generated new: {session_id}")
        user_sessions[session_id] = {
            "conversation": create_langchain_session(),
            "current_product": None
        }

    session_data = user_sessions[session_id]
    conversation = session_data["conversation"]

    # Nếu truy vấn rỗng, trả lời lời chào
    if not user_query.query.strip():
        return {
            "session_id": session_id,
            "response": "Chào bạn! Bạn có thể hỏi tôi bất kỳ điều gì về dược phẩm."
        }

    # Lấy lịch sử hội thoại từ memory (đã được tóm tắt)
    history = conversation.memory.load_memory_variables({}).get("history", "")

    # Kiểm tra xem truy vấn mới có liên quan đến cuộc hội thoại trước không
    if not is_query_related_to_previous(history, user_query.query):
        logger.info("Query is NOT related => resetting memory.")
        conversation.memory.clear()
        # Reset thông tin sản phẩm vì truy vấn mới không liên quan đến sản phẩm đã lưu.
        session_data["current_product"] = None

    # Nếu chưa có sản phẩm hoặc truy vấn không liên quan đến sản phẩm đã lưu thì tìm kiếm sản phẩm mới
    current_product = session_data.get("current_product")
    if current_product is None or not is_query_related_to_product(current_product, user_query.query):
        products = find_similar_products_manual(user_query.query, all_products, top_k=1)
        session_data["current_product"] = products
    else:
        products = current_product

    # Xây dựng prompt cuối cùng kết hợp thông tin truy vấn và sản phẩm
    final_input = f"""
You are a helpful drug sales person answering in Vietnamese.
User query: {user_query.query}

Database matched products: {products}

Please respond politely, highlight the benefits, give relevant advice, pricing, promotions if any, and order link if available.
"""

    # Lấy phản hồi từ conversation chain
    response_text = conversation.predict(input=final_input)
    logger.info("[/chat] Response generated.")

    return {
        "session_id": session_id,
        "response": response_text
    }
