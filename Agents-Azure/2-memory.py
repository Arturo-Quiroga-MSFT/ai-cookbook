"""
Memory: Stores and retrieves relevant information across interactions with Azure OpenAI.
This component maintains conversation history and context to enable coherent multi-turn interactions.

Azure OpenAI Documentation: https://docs.microsoft.com/en-us/azure/cognitive-services/openai/concepts/chat-completions
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential

# Load environment variables
load_dotenv()


def get_azure_openai_client():
    """Initialize Azure OpenAI client with proper authentication."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
    
    if api_key:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    else:
        credential = DefaultAzureCredential()
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=credential,
            api_version=api_version,
        )
    
    return client


def ask_joke_without_memory():
    """Ask for a joke without maintaining conversation history."""
    client = get_azure_openai_client()
    deployment_name = os.getenv("AZURE_OPENAI_GPT4_MINI_DEPLOYMENT", "gpt-4o-mini")
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "user", "content": "Tell me a joke about programming"},
            ],
            temperature=0.8,
            max_tokens=500,
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e}")
        return f"Error: Unable to process request. {str(e)}"


def ask_followup_without_memory():
    """Ask a follow-up question without conversation context."""
    client = get_azure_openai_client()
    deployment_name = os.getenv("AZURE_OPENAI_GPT4_MINI_DEPLOYMENT", "gpt-4o-mini")
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "user", "content": "What was my previous question?"},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e}")
        return f"Error: Unable to process request. {str(e)}"


def ask_followup_with_memory(joke_response: str):
    """Ask a follow-up question with conversation history maintained."""
    client = get_azure_openai_client()
    deployment_name = os.getenv("AZURE_OPENAI_GPT4_MINI_DEPLOYMENT", "gpt-4o-mini")
    
    # Maintain conversation history by including previous messages
    conversation_history = [
        {"role": "user", "content": "Tell me a joke about programming"},
        {"role": "assistant", "content": joke_response},
        {"role": "user", "content": "What was my previous question?"},
    ]
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=conversation_history,
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e}")
        return f"Error: Unable to process request. {str(e)}"


class ConversationMemory:
    """
    A simple conversation memory class to manage chat history.
    In production, you might want to store this in a database or Redis.
    """
    
    def __init__(self):
        self.messages = []
        self.client = get_azure_openai_client()
        self.deployment_name = os.getenv("AZURE_OPENAI_GPT4_MINI_DEPLOYMENT", "gpt-4o-mini")
    
    def add_user_message(self, content: str):
        """Add a user message to the conversation history."""
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str):
        """Add an assistant message to the conversation history."""
        self.messages.append({"role": "assistant", "content": content})
    
    def get_response(self, user_input: str) -> str:
        """
        Get a response from Azure OpenAI while maintaining conversation context.
        
        Args:
            user_input: The user's input message
            
        Returns:
            Assistant's response
        """
        # Add user message to history
        self.add_user_message(user_input)
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=self.messages,
                temperature=0.7,
                max_tokens=1000,
            )
            
            assistant_response = response.choices[0].message.content
            
            # Add assistant response to history
            self.add_assistant_message(assistant_response)
            
            return assistant_response
        
        except Exception as e:
            error_msg = f"Error: Unable to process request. {str(e)}"
            self.add_assistant_message(error_msg)
            return error_msg
    
    def clear_history(self):
        """Clear the conversation history."""
        self.messages = []
    
    def get_message_count(self) -> int:
        """Get the number of messages in the conversation history."""
        return len(self.messages)


if __name__ == "__main__":
    print("=== Memory Demo ===\n")
    
    # First: Ask for a joke without memory
    print("1. Asking for a joke...")
    joke_response = ask_joke_without_memory()
    print(f"Response: {joke_response}\n")

    # Second: Ask follow-up without memory (AI will be confused)
    print("2. Asking follow-up without memory...")
    confused_response = ask_followup_without_memory()
    print(f"Response: {confused_response}\n")

    # Third: Ask follow-up with memory (AI will remember)
    print("3. Asking follow-up with memory...")
    memory_response = ask_followup_with_memory(joke_response)
    print(f"Response: {memory_response}\n")
    
    # Fourth: Demonstrate conversation memory class
    print("4. Using ConversationMemory class...")
    conversation = ConversationMemory()
    
    # Have a multi-turn conversation
    response1 = conversation.get_response("Tell me about Python programming")
    print(f"Response 1: {response1}\n")
    
    response2 = conversation.get_response("What are its main advantages?")
    print(f"Response 2: {response2}\n")
    
    response3 = conversation.get_response("Can you give me a simple example?")
    print(f"Response 3: {response3}\n")
    
    print(f"Total messages in conversation: {conversation.get_message_count()}")
