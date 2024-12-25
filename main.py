from fastapi import FastAPI, Request, Response, Depends
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import openai
from pydantic import BaseModel
from typing import List
import uuid
from fastapi.responses import JSONResponse
from get_db import get_all_products
from similarly import find_similar_products_manual
from call_llm import LLMContentGenerator
from fastapi.middleware.cors import CORSMiddleware

#Get all product from db
# all_products = get_all_products()

# # def find_similar_products_manual(user_query, top_k=5, products):

# products = find_similar_products_manual("Có thuốc giảm mất ngủ không?",all_products,top_k=1, )


# # Khởi tạo ứng dụng FastAPI
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )
# # Lưu trữ session ID cho người dùng
# user_sessions = {}

# # Hàm gọi LLM để tạo phản hồi cho người dùng
# def generate_response(products, user_query):
    
#     # Tạo system prompt và user prompt
#     system_prompt = """
#     You are a helpful salesperson in a pharmacy, answering in Vietnamese. 
#     A customer has asked for advice and you are suggesting suitable products from your inventory.
#     """
    
#     user_prompt = f"""
#     A customer asked: "{user_query}"

#     Based on your inventory, the following products match their query:
#     {products}

#     Please respond to the customer politely and suggest these products, highlighting their benefits.
#     """
    
#     # Gọi LLMContentGenerator để tạo câu trả lời
#     response = LLMContentGenerator().completion(
#         system_prompt=system_prompt,
#         user_prompt=user_prompt,
#         providers=[
#             {
#                 "name": "openai",
#                 "model": "gpt-4o",  # Sử dụng mô hình GPT-4
#                 "retry": 3,
#                 "temperature": 0.2  # Tùy chỉnh độ sáng tạo của mô hình
#             },
#             {
#                 "name": "gemini",
#                 "model": "gemini-1.5-pro",  # Sử dụng mô hình Gemini
#                 "retry": 3,
#                 "temperature": 0.5  # Tùy chỉnh độ sáng tạo của mô hình Gemini
#             }
#         ],
#         json=False  # Trả về kết quả dưới dạng văn bản thay vì JSON
#     )

#     # Trả về câu trả lời từ LLM
#     return response
# # a = generate_response(products,"Có thuốc giảm mất ngủ không?")
# # Model dữ liệu cho request
# class UserQuery(BaseModel):
#     query: str

# # Create a new LangChain session
# def create_langchain_session():
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     chat = ChatOpenAI(model="gpt-4")  # Replace with the desired model
#     conversation = ConversationChain(memory=memory, llm=chat)
#     return conversation

# # Get or create session ID
# def get_or_create_session_id(request: Request):
#     session_id = request.cookies.get("session_id")
#     if not session_id:
#         session_id = str(uuid.uuid4())
#     return session_id

# # Check if a query is related to the previous one
# def is_query_related(conversation, new_query):
#     check_prompt = f"""
#     A customer previously asked: "{conversation.memory.chat_history[-1].content}".
#     They now ask: "{new_query}".
#     Is the new question related to the previous one? Respond with "yes" or "no".
#     """
#     check_response = conversation.predict(input=check_prompt).strip().lower()
#     return check_response == "yes"

# @app.post("/query/")
# async def query_llm(request: Request, user_query: UserQuery, response: Response):
#     session_id = get_or_create_session_id(request)

#     if session_id not in user_sessions:
#         user_sessions[session_id] = create_langchain_session()

#     user_query_text = user_query.query

#     # Query product information
#     product_ids, products = find_similar_products_manual(user_query_text, top_k=1)

#     # Generate response using LLM
#     response_text = generate_response(product_ids, products, user_query_text)

#     response.set_cookie(key="session_id", value=session_id, httponly=True)

#     return {"response": response_text}

# @app.post("/chat/")
# async def chat(request: Request, user_query: UserQuery, response: Response):
#     session_id = get_or_create_session_id(request)

#     if session_id not in user_sessions:
#         user_sessions[session_id] = create_langchain_session()

#     conversation = user_sessions[session_id]
#     new_query = user_query.query

#     # Check if the new query is related to the previous one
#     if len(conversation.memory.chat_history) > 0 and not is_query_related(conversation, new_query):
#         # If unrelated, query products again
#         product_ids, products = find_similar_products_manual(new_query, top_k=1)
#         response_text = generate_response(product_ids, products, new_query)
#     else:
#         # If related, continue the conversation
#         response_text = conversation.predict(input=new_query)

#     response.set_cookie(key="session_id", value=session_id, httponly=True)

#     return {"response": response_text}


# from fastapi import FastAPI, Request, Response, Query
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import os
# import uuid
# import logging

# from langchain_openai import ChatOpenAI
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ----------------------------------------------------------------------
# # Tạm thời list products cho demo
# # ----------------------------------------------------------------------
# # #Get all product from db
# all_products = get_all_products()
# # ----------------------------------------------------------------------
# # Khởi tạo FastAPI
# # ----------------------------------------------------------------------
# app = FastAPI()

# # ----------------------------------------------------------------------
# # Cấu hình CORS:
# #   - Chú ý: khi bật allow_credentials=True, bạn không nên để allow_origins=["*"].
# # ----------------------------------------------------------------------
# frontend_origin = "http://localhost:5500"  # Thay bằng địa chỉ FE của bạn
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[frontend_origin],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ----------------------------------------------------------------------
# # Lưu trữ ConversationChain theo session_id
# # ----------------------------------------------------------------------
# user_sessions = {}

# # ----------------------------------------------------------------------
# # Tạo ConversationChain
# # ----------------------------------------------------------------------
# def create_langchain_session() -> ConversationChain:
#     logger.info("Creating new LangChain session...")
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise ValueError("OPENAI_API_KEY environment variable is not set")

#     llm = ChatOpenAI(
#         temperature=0.7, 
#         openai_api_key=api_key, 
#         model="gpt-4o-mini"
#     )

#     memory = ConversationBufferMemory()
#     prompt = PromptTemplate(
#         input_variables=["history", "input"],
#         template="""
#         Previous conversation:
#         {history}

#         Customer: {input}
#         Assistant:
#         """
#     )

#     conversation = ConversationChain(
#         llm=llm,
#         memory=memory,
#         prompt=prompt,
#         verbose=True
#     )
#     return conversation

# # ----------------------------------------------------------------------
# # Model Input
# # ----------------------------------------------------------------------
# class UserQuery(BaseModel):
#     query: str

# # ----------------------------------------------------------------------
# # Startup Event
# # ----------------------------------------------------------------------
# @app.on_event("startup")
# async def startup_event():
#     if not os.getenv("OPENAI_API_KEY"):
#         logger.error("OPENAI_API_KEY environment variable is not set")
#         raise ValueError("OPENAI_API_KEY environment variable is not set")
#     logger.info("Startup checks completed successfully.")

# # ----------------------------------------------------------------------
# # Endpoint khởi tạo session (GET)
# #   - Ở đây, ta chỉ đơn giản tạo session_id mới và trả về cho FE
# #   - (Không set cookie)
# # ----------------------------------------------------------------------
# @app.get("/init-session")
# async def init_session():
#     session_id = str(uuid.uuid4())
#     logger.info(f"Created new session_id: {session_id}")

#     # Tạo conversation mới
#     user_sessions[session_id] = create_langchain_session()

#     # Trả về JSON kèm session_id để FE hiển thị.
#     return {
#         "session_id": session_id,
#         "detail": "Session initialized"
#     }

# # ----------------------------------------------------------------------
# # Endpoint chat (POST)
# #   - Nhận session_id qua query, ví dụ:
# #       POST /chat?session_id=abc-xyz
# #   - Body: { "query": "nội dung người dùng" }
# # ----------------------------------------------------------------------
# @app.post("/chat/")
# async def chat_endpoint(
#     user_query: UserQuery,
#     session_id: str = Query(None)
# ):
#     logger.info(f"[/chat] Received chat query: {user_query.query}")

#     # Nếu FE chưa truyền session_id, ta tự tạo
#     if not session_id:
#         session_id = str(uuid.uuid4())
#         logger.info(f"No session_id provided in query. Generated new: {session_id}")

#         # Tạo conversation mới
#         user_sessions[session_id] = create_langchain_session()
#     else:
#         # Kiểm tra nếu chưa có trong user_sessions thì tạo mới
#         if session_id not in user_sessions:
#             logger.info(f"No existing conversation for {session_id}, creating new.")
#             user_sessions[session_id] = create_langchain_session()

#     conversation = user_sessions[session_id]

#     # Xử lý query rỗng
#     if not user_query.query.strip():
#         return {
#             "session_id": session_id,
#             "response": "Chào bạn! Bạn có thể hỏi tôi bất kỳ điều gì về dược phẩm."
#         }

#     # ------------------------------------------------------------------
#     # Bước 1: Tìm sản phẩm phù hợp
#     # ------------------------------------------------------------------
#     print(user_query.query)
#     products = find_similar_products_manual(user_query.query,all_products, top_k=1)

#     # ------------------------------------------------------------------
#     # Bước 2: Tạo "input" cuối cùng (gộp system_prompt + product_info + user_query)
#     # ------------------------------------------------------------------
#     # System prompt + user prompt + thông tin product => gộp vào biến input
#     final_input = f"""
#     You are a helpful drug sales person answering customer questions in Vietnamese. Below is the user input and the input after I have queried the database. Please advise on drug sales using the data below. Reply with advice, price, promotions if any and order link
#     Based on your inventory, these products match their query:
#     {products}

#     Please respond politely, highlighting the benefits of these products and giving relevant advice.
#     """

#     # ------------------------------------------------------------------
#     # Bước 3: Gọi LLM (giữ nguyên conversation.predict(input=...) )
#     # ------------------------------------------------------------------
#     response_text = conversation.predict(input=final_input)
#     logger.info("[/chat] Chat response generated successfully")

#     return {
#         "session_id": session_id,
#         "response": response_text
#     }
from fastapi import FastAPI, Request, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import logging

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Giả sử bạn có hàm lấy danh sách product từ DB
all_products = get_all_products()

app = FastAPI()

frontend_origin = "http://localhost:5500"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_sessions = {}

def create_langchain_session() -> ConversationChain:
    logger.info("Creating new LangChain session...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    llm = ChatOpenAI(
        temperature=0.7,
        openai_api_key=api_key,
        model="gpt-4o-mini"
    )

    memory = ConversationBufferMemory()
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""
        Previous conversation:
        {history}

        Customer: {input}
        Assistant:
        """
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
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set")
        raise ValueError("OPENAI_API_KEY is not set")
    logger.info("Startup checks completed successfully.")

@app.get("/init-session")
async def init_session():
    session_id = str(uuid.uuid4())
    logger.info(f"Created new session_id: {session_id}")

    user_sessions[session_id] = create_langchain_session()

    return {
        "session_id": session_id,
        "detail": "Session initialized"
    }

def is_query_related_to_previous(history, query) -> bool:
    logger.info("Checking if query is related to the previous conversation...")
    llm = ChatOpenAI(
        temperature=0.0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
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

@app.post("/chat/")
async def chat_endpoint(
    user_query: UserQuery,
    session_id: str = Query(None)
):
    logger.info(f"[/chat] Received chat query: {user_query.query}")

    # Nếu FE chưa truyền session_id, tự động tạo mới
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"No session_id provided, generated new: {session_id}")
        user_sessions[session_id] = create_langchain_session()

    # Nếu chưa có session_id này, cũng tạo mới
    if session_id not in user_sessions:
        logger.info(f"No existing conversation for {session_id}, creating new.")
        user_sessions[session_id] = create_langchain_session()

    conversation = user_sessions[session_id]

    # Xử lý query rỗng
    if not user_query.query.strip():
        return {
            "session_id": session_id,
            "response": "Chào bạn! Bạn có thể hỏi tôi bất kỳ điều gì về dược phẩm."
        }

    # Lấy history hiện tại
    history = conversation.memory.load_memory_variables({}).get("history", "")

    # Kiểm tra liên quan
    if not is_query_related_to_previous(history, user_query.query):
        logger.info("Query is NOT related => resetting memory.")
        # clear old conversation
        conversation.memory.clear()

    # (1) Tìm sản phẩm phù hợp
    products = find_similar_products_manual(user_query.query, all_products, top_k=1)

    # (2) Tạo final_input
    final_input = f"""
    You are a helpful drug sales person answering in Vietnamese.
    User query: {user_query.query}

    Database matched products: {products}

    Please respond politely, highlight the benefits, give relevant advice, pricing, promotions if any, and order link if available.
    """

    # (3) Gọi conversation.predict
    response_text = conversation.predict(input=final_input)
    logger.info("[/chat] Response generated.")

    return {
        "session_id": session_id,
        "response": response_text
    }

