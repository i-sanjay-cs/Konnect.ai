from langchain_community.llms import DeepInfra
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Ensure API token is set
os.getenv('DEEPINFRA_API_TOKEN')

# Initialize the LLM
llm = DeepInfra(model_id="meta-llama/Meta-Llama-3.1-70B-Instruct")
llm.model_kwargs = {
    "temperature": 0.2,
    "repetition_penalty": 1.2,
    "max_new_tokens": 2000,
    "top_p": 0.9,
}

# Initialize memory for the conversation buffer
memory = ConversationBufferMemory()

# Define the system prompt for the API master
system_prompt = """
You are an API master agent, specializing in API-related queries and best practices. Respond only to questions related to APIs, such as design, security, rate limiting, or other best practices. Always keep your answers concise and easy to understand. Use bullet points, numbers, or headers when explaining complex topics. Limit your response to no more than 3-5 lines per question.
Keep the answers not to much long. As well as if somebody ask anything other then the Api related stuff just tell like "I am a Api Specialist i can't able to answer or something related it."

"""

# Create a conversational chain with memory and the defined system prompt
conversation_chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=PromptTemplate(input_variables=["history", "input"], template=system_prompt + "\n{history}\n\nUser: {input}\nAI:"),
)

def handle_conversation(user_input):
    """
    Function to handle the conversation and return the chatbot's response.
    It uses the conversation chain to maintain context and memory.
    """
    response = conversation_chain.run(input=user_input)
    return response
