"""
Intelligence: The "brain" that processes information and makes decisions using Azure OpenAI LLMs.
This component handles context understanding, instruction following, and response generation.

Azure OpenAI Documentation: https://docs.microsoft.com/en-us/azure/cognitive-services/openai/
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential

# Load environment variables
load_dotenv()


def get_azure_openai_client():
    """
    Initialize Azure OpenAI client with proper authentication.
    
    Uses API key authentication for development and Managed Identity for production.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
    
    # Use API key if available (development), otherwise use Managed Identity (production)
    if api_key:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    else:
        # Use Managed Identity for production environments
        credential = DefaultAzureCredential()
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=credential,
            api_version=api_version,
        )
    
    return client


def basic_intelligence(prompt: str) -> str:
    """
    Basic intelligence function using Azure OpenAI.
    
    Args:
        prompt: The input prompt for the AI model
        
    Returns:
        Generated response text
    """
    client = get_azure_openai_client()
    deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e}")
        return f"Error: Unable to process request. {str(e)}"


if __name__ == "__main__":
    # Test the basic intelligence function
    result = basic_intelligence(prompt="What is artificial intelligence?")
    print("Basic Intelligence Output:")
    print(result)
