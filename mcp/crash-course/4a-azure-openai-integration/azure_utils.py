"""
Azure OpenAI utilities for MCP integration.

This module provides shared utilities for Azure OpenAI client initialization
and configuration management for MCP (Model Context Protocol) integration.
"""

import os
import logging
from typing import Optional, Tuple
from azure.identity import DefaultAzureCredential
from openai import AsyncAzureOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_azure_environment() -> Tuple[bool, str]:
    """
    Validate that required Azure OpenAI environment variables are set.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        return False, error_msg
    
    # Check for authentication method
    has_api_key = bool(os.getenv("AZURE_OPENAI_API_KEY"))
    has_managed_identity = bool(os.getenv("AZURE_CLIENT_ID")) or bool(os.getenv("AZURE_CLIENT_SECRET"))
    
    if not has_api_key and not has_managed_identity:
        error_msg = "No authentication method found. Set AZURE_OPENAI_API_KEY or configure Managed Identity"
        logger.error(error_msg)
        return False, error_msg
    
    return True, ""


def get_azure_openai_client() -> AsyncAzureOpenAI:
    """
    Initialize and return an Azure OpenAI client with proper authentication.
    
    Returns:
        Configured AsyncAzureOpenAI client
        
    Raises:
        ValueError: If required environment variables are missing
        Exception: If client initialization fails
    """
    # Validate environment
    is_valid, error_msg = validate_azure_environment()
    if not is_valid:
        raise ValueError(error_msg)
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    try:
        if api_key:
            # Use API key authentication
            logger.info("Azure OpenAI client initialized with API key authentication")
            return AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2025-01-01-preview"
            )
        else:
            # Use Managed Identity authentication
            logger.info("Azure OpenAI client initialized with Managed Identity authentication")
            credential = DefaultAzureCredential()
            return AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=credential.get_token,
                api_version="2025-01-01-preview"
            )
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
        raise


def get_deployment_name() -> str:
    """
    Get the Azure OpenAI deployment name from environment variables.
    
    Returns:
        Deployment name
        
    Raises:
        ValueError: If deployment name is not configured
    """
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not deployment_name:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME environment variable is required")
    
    return deployment_name


def get_azure_openai_config() -> dict:
    """
    Get complete Azure OpenAI configuration for logging and debugging.
    
    Returns:
        Dictionary with configuration details (excluding sensitive data)
    """
    return {
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "Not set"),
        "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "Not set"),
        "api_version": "2025-01-01-preview",
        "has_api_key": bool(os.getenv("AZURE_OPENAI_API_KEY")),
        "has_managed_identity_config": bool(os.getenv("AZURE_CLIENT_ID")) or bool(os.getenv("AZURE_CLIENT_SECRET"))
    }
