"""
Azure OpenAI MCP Client - Simple Integration

This client demonstrates a streamlined integration between Azure OpenAI models and MCP tools.
It provides a simple, easy-to-understand implementation that's perfect for getting started
with Azure OpenAI and MCP integration.

Key Features:
- Simple Azure OpenAI integration with MCP tools
- Basic error handling
- Support for API key and Managed Identity authentication
- Clean, readable code structure
- Easy to modify and extend
"""

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List

import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from azure_utils import get_azure_openai_client, get_deployment_name, validate_azure_environment

# Apply nest_asyncio to allow nested event loops (needed for Jupyter/IPython)
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store session state
session = None
exit_stack = AsyncExitStack()
azure_openai_client = None
deployment_name = None
stdio = None
write = None


async def initialize_azure_client():
    """Initialize the Azure OpenAI client with proper validation."""
    global azure_openai_client, deployment_name
    
    # Validate environment
    is_valid, error_msg = validate_azure_environment()
    if not is_valid:
        raise ValueError(f"Environment validation failed: {error_msg}")
    
    # Initialize client and get deployment name
    azure_openai_client = get_azure_openai_client()
    deployment_name = get_deployment_name()
    
    logger.info("Azure OpenAI client initialized successfully")


async def connect_to_server(server_script_path: str = "/Users/arturoquiroga/GITHUB/ai-cookbook/mcp/crash-course/4a-azure-openai-integration/server.py"):
    """
    Connect to an MCP server.

    Args:
        server_script_path: Path to the server script.
    """
    global session, stdio, write, exit_stack

    logger.info(f"Connecting to MCP server: {server_script_path}")

    # Server configuration
    server_params = StdioServerParameters(
        command="python",
        args=[server_script_path],
    )

    # Connect to the server
    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))

    # Initialize the connection
    await session.initialize()

    # List available tools
    tools_result = await session.list_tools()
    logger.info("Connected to MCP server successfully")
    logger.info("Available tools:")
    for tool in tools_result.tools:
        logger.info(f"  - {tool.name}: {tool.description}")


async def get_mcp_tools() -> List[Dict[str, Any]]:
    """
    Get available tools from the MCP server in Azure OpenAI format.

    Returns:
        A list of tools in Azure OpenAI function calling format.
    """
    global session

    tools_result = await session.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools_result.tools
    ]


async def process_query(query: str) -> str:
    """
    Process a query using Azure OpenAI and available MCP tools.

    Args:
        query: The user query.

    Returns:
        The response from Azure OpenAI.
    """
    global session, azure_openai_client, deployment_name

    logger.info(f"Processing query: {query[:50]}...")

    # Get available tools
    tools = await get_mcp_tools()

    # Initial Azure OpenAI API call
    response = await azure_openai_client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto",
    )

    # Get assistant's response
    assistant_message = response.choices[0].message

    # Initialize conversation with user query and assistant response
    messages = [
        {"role": "user", "content": query},
        assistant_message,
    ]

    # Handle tool calls if present
    if assistant_message.tool_calls:
        logger.info(f"Processing {len(assistant_message.tool_calls)} tool call(s)")
        
        # Process each tool call
        for tool_call in assistant_message.tool_calls:
            logger.info(f"Executing tool: {tool_call.function.name}")
            
            # Execute tool call
            result = await session.call_tool(
                tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
            )

            # Add tool response to conversation
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.content[0].text,
                }
            )

        # Get final response from Azure OpenAI with tool results
        final_response = await azure_openai_client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            tools=tools,
            tool_choice="none",  # Don't allow more tool calls
        )

        return final_response.choices[0].message.content

    # No tool calls, just return the direct response
    return assistant_message.content


async def cleanup():
    """Clean up resources."""
    global exit_stack
    await exit_stack.aclose()


async def main():
    """Main entry point for the simple Azure OpenAI MCP client."""
    try:
        logger.info("Starting simple Azure OpenAI MCP integration demo")
        
        # Initialize Azure OpenAI client
        await initialize_azure_client()
        
        # Connect to MCP server
        await connect_to_server("server.py")

        # Example: Ask about company vacation policy
        query = "What is our company's vacation policy?"
        print(f"\nQuery: {query}")

        response = await process_query(query)
        print(f"\nResponse: {response}")

        # Additional example: Ask about multiple topics
        query2 = "Can you tell me about our remote work policy and how to submit expense reports?"
        print(f"\n\nQuery: {query2}")

        response2 = await process_query(query2)
        print(f"\nResponse: {response2}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"\nError: {str(e)}")
        print("\nPlease ensure:")
        print("1. Your .env file contains the required Azure OpenAI variables")
        print("2. The MCP server dependencies are installed")
        print("3. The Azure OpenAI service is accessible")
        
    finally:
        await cleanup()


if __name__ == "__main__":
    asyncio.run(main())
