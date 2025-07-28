"""
Recovery: Manages failures and exceptions gracefully in Azure OpenAI agent workflows.
This component implements retry logic, fallback processes, and error handling to ensure system resilience.

Azure OpenAI Error Handling: https://docs.microsoft.com/en-us/azure/cognitive-services/openai/how-to/error-handling
"""

import os
import time
import logging
from typing import Optional, Any, Callable, Union
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from pydantic import BaseModel, ValidationError
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


class UserInfo(BaseModel):
    """User information schema with optional fields for graceful degradation."""
    name: Optional[str] = None
    email: Optional[str] = None
    age: Optional[int] = None
    phone: Optional[str] = None
    company: Optional[str] = None


class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


def exponential_backoff(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay for exponential backoff with optional jitter.
    
    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration
        
    Returns:
        Delay in seconds
    """
    delay = config.initial_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)
    
    if config.jitter:
        import random
        delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
    
    return delay


def retry_with_backoff(
    func: Callable,
    config: RetryConfig = None,
    exceptions: tuple = None
) -> Any:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        config: Retry configuration
        exceptions: Tuple of exceptions to catch and retry on
        
    Returns:
        Function result
    """
    if config is None:
        config = RetryConfig()
    
    if exceptions is None:
        exceptions = (Exception,)
    
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            
            if attempt < config.max_attempts - 1:
                delay = exponential_backoff(attempt, config)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {config.max_attempts} attempts failed. Last error: {e}")
    
    raise last_exception


def resilient_intelligence(prompt: str) -> str:
    """
    Resilient intelligence with comprehensive error handling and fallbacks.
    
    Args:
        prompt: Input prompt
        
    Returns:
        Response with graceful failure handling
    """
    client = get_azure_openai_client()
    deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
    
    def make_request():
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000,
                timeout=30  # 30 second timeout
            )
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Azure OpenAI request failed: {e}")
            raise
    
    try:
        # Attempt with retry logic
        config = RetryConfig(max_attempts=3, initial_delay=1.0)
        return retry_with_backoff(make_request, config)
        
    except Exception as e:
        logger.error(f"All attempts failed: {e}")
        # Fallback response
        return f"I'm experiencing technical difficulties right now. Please try again later. Error: {str(e)[:100]}"


def extract_user_info_with_fallback(prompt: str) -> dict:
    """
    Extract user information with fallback to partial data on validation errors.
    
    Args:
        prompt: Input text containing user information
        
    Returns:
        User information with graceful degradation
    """
    client = get_azure_openai_client()
    deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
    
    system_prompt = """
    Extract user information from the text and return as JSON.
    
    Required JSON format:
    {
        "name": "string or null",
        "email": "string or null", 
        "age": "number or null",
        "phone": "string or null",
        "company": "string or null"
    }
    
    Rules:
    - If a field is not found, use null (not empty string)
    - name: Extract full name if available
    - email: Extract valid email address if available
    - age: Extract age as number if mentioned
    - phone: Extract phone number if available
    - company: Extract company/organization name if mentioned
    
    Example:
    Input: "Hi, I'm Jane"
    Output: {"name": "Jane", "email": null, "age": null, "phone": null, "company": null}
    
    Return only valid JSON.
    """
    
    def extract_attempt():
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        json_data = json.loads(response_text)
        return UserInfo(**json_data)
    
    try:
        # Primary extraction attempt
        user_data = retry_with_backoff(
            extract_attempt,
            RetryConfig(max_attempts=2),
            (json.JSONDecodeError, ValidationError, Exception)
        )
        
        return {
            "success": True,
            "data": user_data.model_dump(),
            "source": "structured_extraction"
        }
        
    except ValidationError as e:
        logger.warning(f"Validation failed, attempting partial extraction: {e}")
        
        try:
            # Fallback: Try to extract just available fields
            simple_prompt = """
            Extract any available information from this text as JSON: """ + prompt + """
            
            Use this format:
            {
                "name": "string or null",
                "email": "string or null"
            }
            
            If no information is found, use null for all fields.
            """
            
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": simple_prompt}],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            partial_data = json.loads(response.choices[0].message.content)
            
            # Create user info with available data
            user_info = UserInfo(
                name=partial_data.get("name"),
                email=partial_data.get("email")
            )
            
            return {
                "success": True,
                "data": user_info.model_dump(),
                "source": "partial_extraction",
                "warning": "Some fields could not be extracted"
            }
        
        except Exception as fallback_error:
            logger.error(f"Fallback extraction also failed: {fallback_error}")
        
        # Final fallback: Return empty user info
        return {
            "success": False,
            "data": {"name": None, "email": None, "age": None, "phone": None, "company": None},
            "source": "fallback",
            "error": str(e)
        }
    
    except Exception as e:
        logger.error(f"Complete extraction failure: {e}")
        return {
            "success": False,
            "data": {"name": None, "email": None, "age": None, "phone": None, "company": None},
            "source": "error_fallback",
            "error": str(e)
        }


class ResilientAgent:
    """
    Agent with comprehensive resilience patterns and error recovery.
    """
    
    def __init__(self, retry_config: RetryConfig = None):
        self.client = get_azure_openai_client()
        self.deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
        self.retry_config = retry_config or RetryConfig()
        self.error_count = 0
        self.success_count = 0
        self.circuit_breaker_threshold = 5  # Open circuit after 5 consecutive failures
        self.circuit_open = False
        self.last_success_time = time.time()
    
    def _should_circuit_break(self) -> bool:
        """Check if circuit breaker should trigger."""
        if self.error_count >= self.circuit_breaker_threshold:
            if time.time() - self.last_success_time > 300:  # 5 minutes
                return True
        return False
    
    def _reset_circuit(self):
        """Reset circuit breaker after successful operation."""
        self.circuit_open = False
        self.error_count = 0
        self.last_success_time = time.time()
    
    def safe_chat(self, prompt: str, fallback_response: str = None) -> dict:
        """
        Safe chat with circuit breaker and comprehensive error handling.
        
        Args:
            prompt: User input
            fallback_response: Custom fallback response
            
        Returns:
            Result dictionary with success status and response
        """
        if self.circuit_open or self._should_circuit_break():
            self.circuit_open = True
            fallback = fallback_response or "Service temporarily unavailable. Please try again later."
            return {
                "success": False,
                "response": fallback,
                "source": "circuit_breaker",
                "error": "Circuit breaker open"
            }
        
        def make_chat_request():
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000,
                timeout=30
            )
            return response.choices[0].message.content
        
        try:
            response = retry_with_backoff(make_chat_request, self.retry_config)
            
            # Success - reset error tracking
            self.success_count += 1
            self._reset_circuit()
            
            return {
                "success": True,
                "response": response,
                "source": "azure_openai",
                "attempts": "multiple" if self.retry_config.max_attempts > 1 else "single"
            }
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Chat request failed after retries: {e}")
            
            fallback = fallback_response or f"I'm having technical difficulties. Please try again. Error: {str(e)[:50]}"
            
            return {
                "success": False,
                "response": fallback,
                "source": "error_fallback",
                "error": str(e),
                "error_count": self.error_count
            }
    
    def get_health_status(self) -> dict:
        """Get current health status of the agent."""
        return {
            "circuit_open": self.circuit_open,
            "error_count": self.error_count,
            "success_count": self.success_count,
            "last_success_time": self.last_success_time,
            "health": "healthy" if not self.circuit_open and self.error_count < 3 else "degraded"
        }


if __name__ == "__main__":
    print("=== Recovery Demo ===\n")
    
    # Test 1: Basic resilient intelligence
    print("1. Testing resilient intelligence...")
    try:
        result = resilient_intelligence("Tell me about machine learning in one sentence")
        print(f"Success: {result[:100]}...\n")
    except Exception as e:
        print(f"Failed: {e}\n")
    
    # Test 2: User info extraction with fallback
    print("2. Testing user info extraction with fallback...")
    test_prompts = [
        "My name is John Smith and my email is john@example.com",  # Complete info
        "Hi, I'm Jane",  # Incomplete info
        "Contact me at invalid-email-format",  # Invalid data
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"   Test {i}: '{prompt}'")
        result = extract_user_info_with_fallback(prompt)
        print(f"   Result: {result}")
        print()
    
    # Test 3: Resilient agent with circuit breaker
    print("3. Testing resilient agent...")
    agent = ResilientAgent(RetryConfig(max_attempts=2, initial_delay=0.5))
    
    # Normal operation
    result1 = agent.safe_chat("What is 2+2?")
    print(f"Chat 1: {result1}")
    
    # With custom fallback
    result2 = agent.safe_chat(
        "Explain quantum computing", 
        fallback_response="Quantum computing is complex. Please consult our documentation."
    )
    print(f"Chat 2: {result2}")
    
    # Check health
    health = agent.get_health_status()
    print(f"Agent health: {health}")
    
    # Test error scenarios (this might fail depending on your setup)
    print(f"\n4. Testing error scenarios...")
    
    # Simulate potential failure by using invalid deployment
    original_deployment = agent.deployment_name
    agent.deployment_name = "invalid-deployment-name"
    
    result3 = agent.safe_chat("This should fail gracefully")
    print(f"Error test result: {result3}")
    
    # Restore original deployment
    agent.deployment_name = original_deployment
    
    final_health = agent.get_health_status()
    print(f"Final health status: {final_health}")
