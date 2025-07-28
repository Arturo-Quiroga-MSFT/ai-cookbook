"""
Tool use with Azure OpenAI.
Demonstrates function calling and tool integration using Azure OpenAI service.
"""

import json
import sys
import os

# Add parent directory to path for azure_utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure_utils import get_azure_openai_client, get_deployment_name, validate_environment
import requests
from pydantic import BaseModel, Field
from typing import Optional

"""
Azure OpenAI Function Calling Documentation:
https://docs.microsoft.com/en-us/azure/cognitive-services/openai/how-to/function-calling
"""


# --------------------------------------------------------------
# Define the tool (function) that we want to call
# --------------------------------------------------------------

def get_weather(latitude: float, longitude: float) -> dict:
    """
    Get current weather for provided coordinates.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        
    Returns:
        dict: Current weather data
    """
    try:
        response = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )
        response.raise_for_status()
        data = response.json()
        return data["current"]
    except Exception as e:
        return {"error": f"Failed to fetch weather data: {str(e)}"}


def call_function(name: str, args: dict) -> dict:
    """
    Execute the specified function with given arguments.
    
    Args:
        name: Function name
        args: Function arguments
        
    Returns:
        dict: Function result
    """
    if name == "get_weather":
        return get_weather(**args)
    else:
        return {"error": f"Unknown function: {name}"}


class WeatherResponse(BaseModel):
    """Structured response for weather information."""
    temperature: Optional[float] = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )


def main():
    """Tool use example with Azure OpenAI."""
    
    # Validate environment configuration
    if not validate_environment():
        print("Environment validation failed. Please check your .env configuration.")
        return
    
    try:
        # Initialize Azure OpenAI client
        client = get_azure_openai_client()
        deployment_name = get_deployment_name("gpt4")
        
        # --------------------------------------------------------------
        # Step 1: Define tools for the model
        # --------------------------------------------------------------
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current temperature for provided coordinates in celsius.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "latitude": {"type": "number"},
                            "longitude": {"type": "number"},
                        },
                        "required": ["latitude", "longitude"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ]
        
        system_prompt = "You are a helpful weather assistant."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What's the weather like in Paris today?"},
        ]
        
        # --------------------------------------------------------------
        # Step 2: Call model with tools defined
        # --------------------------------------------------------------
        
        print("Calling Azure OpenAI with tool definition...")
        completion = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            tools=tools,
            temperature=0.1
        )
        
        print("Model response received.")
        
        # --------------------------------------------------------------
        # Step 3: Check if model wants to call function(s)
        # --------------------------------------------------------------
        
        assistant_message = completion.choices[0].message
        
        if assistant_message.tool_calls:
            print(f"Model wants to call {len(assistant_message.tool_calls)} tool(s)")
            
            # Add assistant message to conversation
            messages.append(assistant_message)
            
            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"Executing function: {function_name}")
                print(f"Arguments: {function_args}")
                
                # Execute the function
                result = call_function(function_name, function_args)
                print(f"Function result: {result}")
                
                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
            
            # --------------------------------------------------------------
            # Step 4: Get final response with function results
            # --------------------------------------------------------------
            
            print("\nGetting final response with function results...")
            completion_2 = client.beta.chat.completions.parse(
                model=deployment_name,
                messages=messages,
                tools=tools,
                response_format=WeatherResponse,
                temperature=0.1
            )
            
            # --------------------------------------------------------------
            # Step 5: Display final structured response
            # --------------------------------------------------------------
            
            final_response = completion_2.choices[0].message.parsed
            
            print("\nFinal Weather Response:")
            print("=" * 40)
            print(f"Temperature: {final_response.temperature}°C")
            print(f"Response: {final_response.response}")
            
        else:
            print("No tool calls were made by the model.")
            print(f"Direct response: {assistant_message.content}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your Azure OpenAI configuration is correct in .env file")


if __name__ == "__main__":
    main()
