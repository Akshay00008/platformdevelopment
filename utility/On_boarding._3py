import os
import getpass
import logging
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from utility.retrain_bot import fetch_data
from Databases.mongo import Bot_Retrieval, company_Retrieval


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

try:
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI:")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    api = os.getenv('OPENAI_API_KEY')
    model = os.getenv('GPT_model')
    model_provider = os.getenv('GPT_model_provider')
    if not api or not model or not model_provider:
        raise ValueError("Please check the OPENAI_API_KEY, GPT_model, and GPT_model_provider.")

except Exception as e:
    logger.error(f"Initialization failed: {e}")
    raise


def load_llm(key, model_provider, model_name):
    try:
        if not all([key, model_provider, model_name]):
            raise ValueError("Missing LLM configuration in secrets.")
        os.environ["API_KEY"] = key
        return init_chat_model(model_name, model_provider=model_provider)
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise


llm = load_llm(api, model_provider, model)


# Conversation history per user
converstation_state = {}

# Cache for retrieved FAISS documents: key -> retrieved docs
# Key formed from (chatbot_id, version_id, lowercase question)
retrieval_cache = {}

# Cache for LLM responses: key -> response string
# Key formed from (user_id, chatbot_id, version_id, lowercase question)
llm_response_cache = {}


def chatbot(chatbot_id, version_id, prompt, user_id):
    try:
        request_body = {
            "chatbot_id": chatbot_id,
            "version_id": version_id,
            "collection_name": ["guidance", "handoff"]
        }

        guidelines = fetch_data(request_body)
        print("Guidelines generated successfully")

        Bot_information = Bot_Retrieval(chatbot_id, version_id)
        if not Bot_information:
            pass
           

        bot_company = company_Retrieval()
        if not bot_company:
            pass
        # return "No company information found."

        # Initialize conversation history if needed
        if user_id not in converstation_state:
            converstation_state[user_id] = [{'role': 'user', 'content': prompt}]

        converstation_history = converstation_state[user_id]
        converstation_state[user_id].append({'role': 'user', 'content': prompt})

        greeting = Bot_information[0].get('greeting_message', "Hello!")
        
        purpose = Bot_information[0].get('purpose', "You are an AI assistant helping users with their queries on behalf of the organization. "
    "You provide clear and helpful responses while avoiding personal details (such as name, age, birth information, or contact info) "
    "and sensitive transactional data (such as order details, delivery details, or payment information")
        
        languages = Bot_information[0].get('supported_languages', ["English"])

        tone_and_style = Bot_information[0].get('tone_style', "Friendly and professional")
        
        company_info = bot_company[0].get('bot_company', " You are an AI assistant representing the organization. "
    "Your task is to help customers with their needs and guide them with relevant information about the organization's services, "
    "without disclosing personal user data or sensitive company records.")
        
        print("Received bot info")

        # First check if we have a cached LLM response for this user, bot, version, and prompt
        cache_key_llm = (user_id, chatbot_id, version_id, prompt.lower())
        if cache_key_llm in llm_response_cache:
            logger.info("Returning cached LLM response")
            return llm_response_cache[cache_key_llm]

        # Call Personal_chatbot with caching-enabled retrieval
        llm_response = Personal_chatbot(converstation_history, prompt, languages, purpose, tone_and_style,
                                       greeting, guidelines, company_info, chatbot_id, version_id)

        # Cache the LLM response
        llm_response_cache[cache_key_llm] = llm_response

        converstation_state[user_id].append({'role': 'bot', 'content': llm_response})

        return llm_response

    except Exception as e:
        logger.error(f"Error in chatbot function: {e}")
        return f"An error occurred: {e}"


def Personal_chatbot(converstation_history, prompt, languages, purpose, tone_and_style, greeting,
                     guidelines, company_info, chatbot_id, version_id):
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def retrieve(state: State):
        # Use cache key based on chatbot, version, and question
        cache_key = (chatbot_id, version_id, state['question'].lower())
        if cache_key in retrieval_cache:
            logger.info("Using cached retrieval results")
            return {"context": retrieval_cache[cache_key]}

        try:
            faiss_index_dir = "/home/bramhesh_srivastav/platformdevelopment/faiss_indexes"

            faiss_index_filename = f"{chatbot_id}_{version_id}_faiss_index"
            faiss_index_website = f"{chatbot_id}_{version_id}_faiss_index_website"

            faiss_path_1 = os.path.join(faiss_index_dir, faiss_index_filename)
            faiss_path_2 = os.path.join(faiss_index_dir, faiss_index_website)

            if not os.path.exists(faiss_path_1):
                raise FileNotFoundError(f"FAISS index not found at {faiss_path_1}")
            if not os.path.exists(faiss_path_2):
                raise FileNotFoundError(f"FAISS index not found at {faiss_path_2}")

            logger.info(f"Loading FAISS index from {faiss_path_1}")
            new_vector_store = FAISS.load_local(faiss_path_1, embeddings, allow_dangerous_deserialization=True)

            logger.info(f"Loading FAISS index from {faiss_path_2}")
            new_vector_store_1 = FAISS.load_local(faiss_path_2, embeddings, allow_dangerous_deserialization=True)

            logger.info("FAISS indexes loaded successfully.")

            retrieved_docs = new_vector_store.similarity_search(state['question'])
            retrieved_docs_2 = new_vector_store_1.similarity_search(state['question'])

            logger.info("retrieved_docs indexes completed successfully.")

            combined_docs = []
            seen_docs = set()

            for doc in retrieved_docs:
                if doc.id not in seen_docs:
                    combined_docs.append(doc)
                    seen_docs.add(doc.id)

            for doc in retrieved_docs_2:
                if doc.id not in seen_docs:
                    combined_docs.append(doc)
                    seen_docs.add(doc.id)

            logger.info("combined_docs returned successfully.")

            # Cache results for future calls
            retrieval_cache[cache_key] = combined_docs

            return {"context": combined_docs}

        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return {"context": []}

    def generate(state: State):
        try:
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = [
                SystemMessage(
                    f"""
                    Role: You are a personal chatbot with the following purpose: {purpose}.
                    You can communicate fluently in the following languages: {languages}.
                    {greeting} Always keep the conversation context in mind, including the chat history:
                    {converstation_history}
                    Do not respond to the query or question which is not related to the company: {company_info}
                    You also have access to context derived from document scores:
                    {docs_content}
                    Maintain a tone and style that aligns with the following guidelines: {guidelines}
                    Maintain a tone and style: {tone_and_style}
                    --When user asks about "delivery status" or "Order status" or 
                    --"Warranty and Service Requests or Order and Shipping Queries" or 
                    --"Specialized Product Configurations[example : Schematic diagrams or wiring diagram]" or 
                    --"Payment or Billing Issues" or
                    --"if user says not working " or 
                    --"user says These words [\"fire\", \"melt\", \"burned\", \"melted\", \"burned up\"] and add synonyms of these or related words" or 
                    --"If user asks about the new product or product not present in your list" or
                    --"if you find the words like [price, pricing, burned up] word in user query" or 
                    --"When the user explicitly asks to speak with a live agent or mentions they are unable to resolve their issue with the chatbot alone" 
                    -- For the above cases please respond with: "Let's get you connected to one of our live agents so they can assist you further. Would it be okay if I connect you now?"
                    """
                ),
                HumanMessage(f"{state['question']}")
            ]
            response = llm.invoke(messages)
            return {"answer": response.content}

        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            return {"answer": "Sorry, something went wrong in generating a response."}

    try:
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        response = graph.invoke({"question": prompt})

        return response.get('answer', "No response generated.")
    except Exception as e:
        logger.error(f"Error in conversation graph: {e}")
        return f"An error occurred during conversation: {e}"
