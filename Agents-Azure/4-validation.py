"""
Validation: Ensures LLM outputs match predefined data schemas using Azure OpenAI.
This component provides schema validation and structured data parsing to guarantee consistent data formats for downstream code.

Azure OpenAI Structured Outputs: https://docs.microsoft.com/en-us/azure/cognitive-services/openai/how-to/structured-outputs
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
import json

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


class TaskResult(BaseModel):
    """
    Task information schema with validation.
    
    More info: https://docs.pydantic.dev
    """
    task: str = Field(description="Description of the task")
    completed: bool = Field(description="Whether the task is completed")
    priority: int = Field(ge=1, le=5, description="Priority level from 1 (low) to 5 (high)")
    due_date: Optional[str] = Field(None, description="Due date in YYYY-MM-DD format")
    category: Optional[str] = Field(None, description="Task category")


class PersonInfo(BaseModel):
    """
    Person information schema with validation.
    """
    name: str = Field(description="Full name of the person")
    age: Optional[int] = Field(None, ge=0, le=150, description="Age in years")
    email: Optional[str] = Field(None, description="Email address")
    occupation: Optional[str] = Field(None, description="Job title or profession")
    skills: List[str] = Field(default_factory=list, description="List of skills")


class ProjectPlan(BaseModel):
    """
    Project plan schema with nested validation.
    """
    project_name: str = Field(description="Name of the project")
    description: str = Field(description="Project description")
    tasks: List[TaskResult] = Field(description="List of tasks in the project")
    estimated_duration_days: int = Field(ge=1, description="Estimated duration in days")
    budget: Optional[float] = Field(None, ge=0, description="Project budget")


def structured_intelligence(prompt: str, schema_class: BaseModel) -> BaseModel:
    """
    Extract structured information using Azure OpenAI with schema validation.
    
    Args:
        prompt: Input prompt for extraction
        schema_class: Pydantic model class for validation
        
    Returns:
        Validated structured data
    """
    client = get_azure_openai_client()
    deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
    
    # Create JSON schema from Pydantic model
    json_schema = schema_class.model_json_schema()
    
    # System prompt for structured extraction
    system_prompt = f"""
    Extract information from the user input and return it as valid JSON that matches this schema:
    
    {json.dumps(json_schema, indent=2)}
    
    Important:
    - Return only valid JSON
    - Follow the schema exactly
    - If information is missing, use null for optional fields
    - Ensure all required fields are present
    """
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,  # Low temperature for consistent extraction
            max_tokens=1500,
            response_format={"type": "json_object"}  # Ensure JSON output
        )
        
        # Parse and validate the response
        response_text = response.choices[0].message.content
        json_data = json.loads(response_text)
        
        # Validate against schema
        validated_data = schema_class(**json_data)
        return validated_data
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        raise ValueError(f"Invalid JSON returned from Azure OpenAI: {e}")
    
    except ValidationError as e:
        print(f"Validation error: {e}")
        raise ValueError(f"Data doesn't match schema: {e}")
    
    except Exception as e:
        print(f"Error in structured extraction: {e}")
        raise ValueError(f"Extraction failed: {e}")


def extract_with_retry(prompt: str, schema_class: BaseModel, max_retries: int = 3) -> BaseModel:
    """
    Extract structured data with retry logic for validation failures.
    
    Args:
        prompt: Input prompt
        schema_class: Pydantic model class
        max_retries: Maximum number of retry attempts
        
    Returns:
        Validated structured data
    """
    client = get_azure_openai_client()
    deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
    
    json_schema = schema_class.model_json_schema()
    
    system_prompt = f"""
    Extract information from the user input and return it as valid JSON that matches this schema:
    
    {json.dumps(json_schema, indent=2)}
    
    Rules:
    - Return only valid JSON
    - Follow the schema exactly
    - If information is missing, use null for optional fields
    - Ensure all required fields are present
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            json_data = json.loads(response_text)
            validated_data = schema_class(**json_data)
            
            return validated_data
            
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                # Add error feedback for retry
                error_message = f"The previous response had an error: {str(e)}. Please fix the JSON format and ensure it matches the schema exactly."
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": error_message})
            else:
                raise ValueError(f"Failed to extract valid data after {max_retries} attempts: {e}")
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise


class ValidationAgent:
    """
    Agent that specializes in structured data extraction and validation.
    """
    
    def __init__(self):
        self.client = get_azure_openai_client()
        self.deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
    
    def extract(self, prompt: str, schema_class: BaseModel, max_retries: int = 3) -> BaseModel:
        """
        Extract and validate structured data.
        
        Args:
            prompt: Input text to extract from
            schema_class: Pydantic model for validation
            max_retries: Maximum retry attempts
            
        Returns:
            Validated structured data
        """
        return extract_with_retry(prompt, schema_class, max_retries)
    
    def validate_json(self, json_string: str, schema_class: BaseModel) -> BaseModel:
        """
        Validate a JSON string against a schema.
        
        Args:
            json_string: JSON string to validate
            schema_class: Pydantic model for validation
            
        Returns:
            Validated data object
        """
        try:
            json_data = json.loads(json_string)
            return schema_class(**json_data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Validation failed: {e}")


if __name__ == "__main__":
    print("=== Validation Demo ===\n")
    
    # Test 1: Task extraction
    print("1. Testing task extraction...")
    task_prompt = "I need to complete the project presentation by Friday, it's high priority and falls under the work category"
    
    try:
        task_result = structured_intelligence(task_prompt, TaskResult)
        print("Task extraction successful:")
        print(task_result.model_dump_json(indent=2))
        print(f"Extracted task: {task_result.task}")
        print(f"Priority: {task_result.priority}")
        print()
    except Exception as e:
        print(f"Task extraction failed: {e}\n")
    
    # Test 2: Person information extraction
    print("2. Testing person information extraction...")
    person_prompt = "Hi, I'm John Doe, 30 years old, software engineer at Tech Corp. My email is john@example.com and I'm skilled in Python, JavaScript, and machine learning."
    
    try:
        person_result = structured_intelligence(person_prompt, PersonInfo)
        print("Person extraction successful:")
        print(person_result.model_dump_json(indent=2))
        print()
    except Exception as e:
        print(f"Person extraction failed: {e}\n")
    
    # Test 3: Complex project plan extraction
    print("3. Testing project plan extraction...")
    project_prompt = """
    We need to create a mobile app for our restaurant. The project includes:
    - Design the user interface (high priority, due 2024-02-15)
    - Develop the backend API (high priority, due 2024-02-28) 
    - Implement user authentication (medium priority, due 2024-03-10)
    - Add payment integration (high priority, due 2024-03-15)
    - Test the application (medium priority, due 2024-03-20)
    
    The estimated duration is 60 days with a budget of $50,000.
    """
    
    try:
        project_result = structured_intelligence(project_prompt, ProjectPlan)
        print("Project extraction successful:")
        print(project_result.model_dump_json(indent=2))
        print()
    except Exception as e:
        print(f"Project extraction failed: {e}\n")
    
    # Test 4: Validation agent with retry
    print("4. Testing validation agent with retry...")
    agent = ValidationAgent()
    
    try:
        retry_result = agent.extract(
            "Create a task to review documentation, low priority, due next Monday",
            TaskResult,
            max_retries=3
        )
        print("Retry extraction successful:")
        print(retry_result.model_dump_json(indent=2))
    except Exception as e:
        print(f"Retry extraction failed: {e}")
