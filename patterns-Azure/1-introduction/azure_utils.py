"""
Shared utilities for Azure OpenAI workflows.
Provides common functionality for authentication, client setup, and error handling.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_azure_openai_client() -> AzureOpenAI:
    """
    Initialize Azure OpenAI client with proper authentication.
    
    Returns:
        AzureOpenAI: Configured Azure OpenAI client
        
    Raises:
        ValueError: If required environment variables are missing
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
    
    try:
        if api_key:
            # Use API key authentication
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
            logger.info("Azure OpenAI client initialized with API key authentication")
        else:
            # Use Azure Managed Identity
            credential = DefaultAzureCredential()
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=credential,
                api_version=api_version,
            )
            logger.info("Azure OpenAI client initialized with Managed Identity authentication")
        
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {e}")
        raise


def get_deployment_name(model_type: str = "gpt4") -> str:
    """
    Get deployment name for specified model type.
    
    Args:
        model_type: Type of model ('gpt4' or 'gpt35')
        
    Returns:
        str: Deployment name
    """
    if model_type.lower() == "gpt4":
        return os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
    elif model_type.lower() == "gpt35":
        return os.getenv("AZURE_OPENAI_GPT35_DEPLOYMENT", "gpt-35-turbo")
    else:
        return os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")


def validate_environment() -> bool:
    """
    Validate that required environment variables are set.
    
    Returns:
        bool: True if environment is properly configured
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    if not endpoint:
        logger.error("AZURE_OPENAI_ENDPOINT environment variable is required")
        return False
    
    if not api_key:
        logger.warning("AZURE_OPENAI_API_KEY not found. Will attempt Managed Identity authentication.")
    
    return True
