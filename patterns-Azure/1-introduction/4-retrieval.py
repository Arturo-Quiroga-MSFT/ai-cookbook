"""
Retrieval with Azure OpenAI.
Demonstrates knowledge base retrieval and question answering using Azure OpenAI service.
"""

import json
import sys
import os

# Add parent directory to path for azure_utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure_utils import get_azure_openai_client, get_deployment_name, validate_environment
from pydantic import BaseModel, Field
from typing import Optional

"""
Azure OpenAI Function Calling Documentation:
https://docs.microsoft.com/en-us/azure/cognitive-services/openai/how-to/function-calling
"""


# --------------------------------------------------------------
# Define the knowledge base retrieval tool
# --------------------------------------------------------------

def search_kb(question: str) -> dict:
    """
    Load the knowledge base from the JSON file and search for relevant information.
    (This is a simplified example - in production, you'd implement proper search/retrieval)
    
    Args:
        question: User's question
        
    Returns:
        dict: Knowledge base records
    """
    try:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        kb_path = os.path.join(current_dir, "kb.json")
        
        with open(kb_path, "r") as f:
            kb_data = json.load(f)
        
        print(f"Knowledge base loaded with {len(kb_data['records'])} records")
        return kb_data
        
    except Exception as e:
        return {"error": f"Failed to load knowledge base: {str(e)}"}


def call_function(name: str, args: dict) -> dict:
    """
    Execute the specified function with given arguments.
    
    Args:
        name: Function name
        args: Function arguments
        
    Returns:
        dict: Function result
    """
    if name == "search_kb":
        return search_kb(**args)
    else:
        return {"error": f"Unknown function: {name}"}


class KBResponse(BaseModel):
    """Structured response for knowledge base queries."""
    answer: str = Field(description="The answer to the user's question.")
    source: Optional[int] = Field(description="The record id of the answer source.")
    confidence: Optional[float] = Field(description="Confidence in the answer (0-1).")


def ask_question(client, deployment_name: str, question: str, tools: list) -> KBResponse:
    """
    Ask a question and get a structured response.
    
    Args:
        client: Azure OpenAI client
        deployment_name: Model deployment name
        question: User's question
        tools: Available tools
        
    Returns:
        KBResponse: Structured response
    """
    system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    print(f"\nAsking: '{question}'")
    
    # --------------------------------------------------------------
    # Step 1: Call model with tools defined
    # --------------------------------------------------------------
    
    completion = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        tools=tools,
        temperature=0.1
    )
    
    assistant_message = completion.choices[0].message
    
    # --------------------------------------------------------------
    # Step 2: Check if model wants to call function(s)
    # --------------------------------------------------------------
    
    if assistant_message.tool_calls:
        print("Model is searching knowledge base...")
        
        # Add assistant message to conversation
        messages.append(assistant_message)
        
        # Execute each tool call
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Execute the function
            result = call_function(function_name, function_args)
            
            # Add tool result to conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
        
        # --------------------------------------------------------------
        # Step 3: Get final structured response
        # --------------------------------------------------------------
        
        completion_2 = client.beta.chat.completions.parse(
            model=deployment_name,
            messages=messages,
            tools=tools,
            response_format=KBResponse,
            temperature=0.1
        )
        
        return completion_2.choices[0].message.parsed
    
    else:
        # No tool call needed - direct response
        print("Model responding directly (no KB search needed)")
        return KBResponse(
            answer=assistant_message.content,
            source=None,
            confidence=0.8
        )


def main():
    """Retrieval example with Azure OpenAI."""
    
    # Validate environment configuration
    if not validate_environment():
        print("Environment validation failed. Please check your .env configuration.")
        return
    
    try:
        # Initialize Azure OpenAI client
        client = get_azure_openai_client()
        deployment_name = get_deployment_name("gpt4")
        
        # --------------------------------------------------------------
        # Define tools for the model
        # --------------------------------------------------------------
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_kb",
                    "description": "Get the answer to the user's question from the knowledge base about our e-commerce store.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                        },
                        "required": ["question"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ]
        
        print("Azure OpenAI Knowledge Base Assistant")
        print("=" * 50)
        
        # --------------------------------------------------------------
        # Test 1: Question that should trigger KB search
        # --------------------------------------------------------------
        
        kb_questions = [
            "What is the return policy?",
            "Do you ship internationally?",
            "What payment methods do you accept?",
            "How do I track my order?"
        ]
        
        for question in kb_questions:
            response = ask_question(client, deployment_name, question, tools)
            
            print(f"Answer: {response.answer}")
            if response.source:
                print(f"Source: Record ID {response.source}")
            if response.confidence:
                print(f"Confidence: {response.confidence:.2f}")
            print("-" * 30)
        
        # --------------------------------------------------------------
        # Test 2: Question that doesn't trigger the tool
        # --------------------------------------------------------------
        
        print("\nTesting question outside knowledge base scope:")
        off_topic_response = ask_question(
            client, deployment_name, 
            "What is the weather in Tokyo?", 
            tools
        )
        
        print(f"Answer: {off_topic_response.answer}")
        if off_topic_response.confidence:
            print(f"Confidence: {off_topic_response.confidence:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your Azure OpenAI configuration is correct in .env file")


if __name__ == "__main__":
    main()
