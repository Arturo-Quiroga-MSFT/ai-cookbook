"""
Tools: Enables agents to execute specific actions in external systems using Azure OpenAI.
This component provides the capability to make API calls, database updates, file operations, and other practical actions.

Azure OpenAI Function Calling: https://docs.microsoft.com/en-us/azure/cognitive-services/openai/how-to/function-calling
"""

import json
import os
import requests
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


def get_weather(latitude: float, longitude: float) -> dict:
    """
    Get current weather information for provided coordinates.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        
    Returns:
        Weather data including temperature and wind speed
    """
    try:
        response = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "temperature": data["current"]["temperature_2m"],
            "wind_speed": data["current"]["wind_speed_10m"],
            "status": "success"
        }
    except requests.RequestException as e:
        return {
            "error": f"Failed to fetch weather data: {str(e)}",
            "status": "error"
        }


def get_city_coordinates(city: str) -> dict:
    """
    Get latitude and longitude for a city name.
    
    Args:
        city: Name of the city
        
    Returns:
        Coordinates data with latitude and longitude
    """
    try:
        # Using a free geocoding service
        response = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get("results"):
            result = data["results"][0]
            return {
                "latitude": result["latitude"],
                "longitude": result["longitude"],
                "name": result["name"],
                "country": result.get("country", ""),
                "status": "success"
            }
        else:
            return {
                "error": f"City '{city}' not found",
                "status": "error"
            }
    except requests.RequestException as e:
        return {
            "error": f"Failed to fetch city coordinates: {str(e)}",
            "status": "error"
        }


def call_function(name: str, args: dict):
    """
    Execute the specified function with given arguments.
    
    Args:
        name: Function name to call
        args: Arguments to pass to the function
        
    Returns:
        Function execution result
    """
    if name == "get_weather":
        return get_weather(**args)
    elif name == "get_city_coordinates":
        return get_city_coordinates(**args)
    else:
        raise ValueError(f"Unknown function: {name}")


def intelligence_with_tools(prompt: str) -> str:
    """
    Process user input with tool calling capabilities using Azure OpenAI.
    
    Args:
        prompt: User input that may require tool usage
        
    Returns:
        Assistant response after potentially using tools
    """
    client = get_azure_openai_client()
    deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")

    # Define available tools/functions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information for provided coordinates in celsius.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "number",
                            "description": "Latitude coordinate"
                        },
                        "longitude": {
                            "type": "number", 
                            "description": "Longitude coordinate"
                        },
                    },
                    "required": ["latitude", "longitude"],
                    "additionalProperties": False,
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_city_coordinates",
                "description": "Get latitude and longitude coordinates for a city name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Name of the city"
                        },
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            }
        }
    ]

    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant that can get weather information for cities. When a user asks about weather in a city, first get the coordinates for that city, then get the weather for those coordinates. Always provide a complete response with the weather information."
        },
        {"role": "user", "content": prompt}
    ]

    try:
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Call Azure OpenAI with tools
            response = client.chat.completions.create(
                model=deployment_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1500,
            )

            response_message = response.choices[0].message
            
            # Check if the model wants to call any tools
            if response_message.tool_calls:
                # Add the assistant's response to messages (ensure content is not null)
                assistant_message = {
                    "role": "assistant",
                    "content": response_message.content or "",
                    "tool_calls": response_message.tool_calls
                }
                messages.append(assistant_message)
                
                # Execute each tool call
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"Calling function: {function_name} with args: {function_args}")
                    
                    # Execute the function
                    function_result = call_function(function_name, function_args)
                    print(f"Function result: {function_result}")
                    
                    # Add function result to messages (ensure content is string)
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(function_result) if function_result is not None else "{}"
                    })
                
                # Continue the loop to potentially make more tool calls
                continue
            else:
                # No more tools needed, return the response
                return response_message.content or "No response generated"
        
        # If we've reached max iterations, make a final call without tools
        final_response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1500,
        )
        
        return final_response.choices[0].message.content or "No response generated"

    except Exception as e:
        print(f"Error in tool calling: {e}")
        return f"Error: Unable to process request with tools. {str(e)}"


class ToolAgent:
    """
    A more sophisticated agent class that manages tools and conversation state.
    """
    
    def __init__(self):
        self.client = get_azure_openai_client()
        self.deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
        self.messages = []
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information for provided coordinates.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "latitude": {"type": "number"},
                            "longitude": {"type": "number"},
                        },
                        "required": ["latitude", "longitude"],
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_city_coordinates",
                    "description": "Get coordinates for a city name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                }
            }
        ]
    
    def chat(self, user_input: str) -> str:
        """
        Chat with the agent, automatically handling tool calls.
        
        Args:
            user_input: User's message
            
        Returns:
            Agent's response
        """
        self.messages.append({"role": "user", "content": user_input})
        
        try:
            max_iterations = 5  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=self.messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=1500,
                )
                
                response_message = response.choices[0].message
                
                if response_message.tool_calls:
                    # Add assistant message with tool calls (ensure content is not null)
                    assistant_message_with_tools = {
                        "role": "assistant",
                        "content": response_message.content or "",
                        "tool_calls": response_message.tool_calls
                    }
                    self.messages.append(assistant_message_with_tools)
                    
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        function_result = call_function(function_name, function_args)
                        
                        self.messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(function_result) if function_result is not None else "{}"
                        })
                    
                    # Continue the loop to potentially make more tool calls
                    continue
                else:
                    # No more tools needed, add the final response and return
                    assistant_message = response_message.content or "No response generated"
                    self.messages.append({"role": "assistant", "content": assistant_message})
                    return assistant_message
            
            # If we've reached max iterations, make a final call without tools
            final_response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=self.messages,
                temperature=0.7,
                max_tokens=1500,
            )
            
            assistant_message = final_response.choices[0].message.content or "No response generated"
            self.messages.append({"role": "assistant", "content": assistant_message})
            return assistant_message
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.messages.append({"role": "assistant", "content": error_msg})
            return error_msg


if __name__ == "__main__":
    print("=== Tools Demo ===\n")
    
    # Test 1: Simple tool calling
    print("1. Testing weather lookup for Paris...")
    result1 = intelligence_with_tools("What's the weather like in Paris today?")
    print(f"Result: {result1}\n")
    
    # Test 2: Multi-step tool calling (city lookup + weather)
    print("2. Testing weather lookup for Tokyo...")
    result2 = intelligence_with_tools("Can you tell me the current weather in Tokyo, Japan?")
    print(f"Result: {result2}\n")
    
    # Test 3: Using the ToolAgent class
    print("3. Testing ToolAgent conversation...")
    agent = ToolAgent()
    
    response1 = agent.chat("What's the weather like in Mexico city?")
    print(f"Agent: {response1}\n")
    
    response2 = agent.chat("How about in Toronto?")
    print(f"Agent: {response2}\n")
    
    response3 = agent.chat("Which city has better weather?")
    print(f"Agent: {response3}")
