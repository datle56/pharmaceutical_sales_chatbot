# app/llm/prompts.py
from langchain.prompts import PromptTemplate

# Prompt tóm tắt chỉ giữ lại các thông tin quan trọng của cuộc hội thoại.
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["chat_history"],
    template=(
        "Hãy tóm tắt lại những thông tin quan trọng liên quan đến sản phẩm và yêu cầu của khách hàng. "
        "Chỉ giữ lại các chi tiết cần thiết để hiểu ngữ cảnh, không lưu toàn bộ cuộc hội thoại:\n\n{chat_history}"
    )
)

# Prompt cho conversation chain chính.
CONVERSATION_PROMPT = PromptTemplate(
    input_variables=["history", "input", "product_recommendations"],
    template=(
        "Previous conversation summary:\n{history}\n\n"
        "Customer: {input}\n"
        "Product recommendations:\n{product_recommendations}\n"
        "Assistant:"
    )
)

PRODUCT_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "product_recommendations": {
            "type": "array",
            "items": {"type": "string"}
        },
        "additional_info": {"type": "string"}
    },
    "required": ["answer", "product_recommendations"]
}

QUERY_RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_related": {"type": "boolean"}
    },
    "required": ["is_related"]
}
