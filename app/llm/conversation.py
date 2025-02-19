import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from app.config import OPENAI_API_KEY
from app.llm.prompts import SUMMARY_PROMPT, CONVERSATION_PROMPT

logger = logging.getLogger(__name__)

def create_langchain_session():
    """Create a new conversation session using the latest LangChain patterns."""
    logger.info("Creating new LangChain session...")

    try:
        llm = ChatOpenAI(
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY,
            model="gpt-4o-mini"  
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant specializing in drug recommendations."),
            ("human", "{input}")
        ])

        chain = prompt | llm
        message_history = ChatMessageHistory()

        # Create a simple runnable that always uses the same message history
        runnable_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: message_history,
            input_messages_key="input",
            history_messages_key="history"
        )

        logger.info("Successfully created LangChain session")
        return {
            "runnable": runnable_with_history,
            "history": message_history
        }

    except Exception as e:
        logger.error(f"Error creating LangChain session: {str(e)}")
        raise

def get_chat_history(message_history):
    """Convert chat history to string format"""
    logger.info("Converting chat history to string format...")
    messages = []
    for msg in message_history.messages:
        if isinstance(msg, HumanMessage):
            messages.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            messages.append(f"Assistant: {msg.content}")
    return "\n".join(messages)
