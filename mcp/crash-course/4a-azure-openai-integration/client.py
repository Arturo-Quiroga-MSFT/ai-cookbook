"""
Azure OpenAI MCP Client - Advanced Integration

This client demonstrates advanced integration between Azure OpenAI models and MCP tools.
It provides a robust, production-ready implementation with comprehensive error handling,
logging, and support for both API key and Managed Identity authentication.

Key Features:
- Azure OpenAI integration with MCP tools
- Comprehensive error handling and retry logic
- Support for multiple authentication methods
- Detailed logging and monitoring
- Clean resource management
- Conversation state management
"""

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from azure_utils import get_azure_openai_client, get_deployment_name, validate_azure_environment, get_azure_openai_config

# Apply nest_asyncio to allow nested event loops (needed for Jupyter/IPython)
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AzureOpenAIMCPClient:
    """
    Advanced client for interacting with Azure OpenAI models using MCP tools.
    
    This client provides a comprehensive integration between Azure OpenAI and MCP,
    with support for multiple authentication methods, robust error handling,
    and detailed logging.
    """

    def __init__(self, deployment_name: Optional[str] = None):
        """
        Initialize the Azure OpenAI MCP client.

        Args:
            deployment_name: The Azure OpenAI deployment name. If None, will use environment variable.
        """
        # Validate environment first
        is_valid, error_msg = validate_azure_environment()
        if not is_valid:
            raise ValueError(f"Environment validation failed: {error_msg}")
        
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.azure_openai_client = None
        self.deployment_name = deployment_name or get_deployment_name()
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None
        
        # Initialize Azure OpenAI client
        try:
            self.azure_openai_client = get_azure_openai_client()
            logger.info("Azure OpenAI MCP Client initialized successfully")
            
            # Log configuration for debugging
            config = get_azure_openai_config()
            logger.info(f"Configuration: {config}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise

    async def connect_to_server(self, server_script_path: str = "server.py"):
        """
        Connect to an MCP server with comprehensive error handling.

        Args:
            server_script_path: Path to the server script.
            
        Raises:
            Exception: If server connection fails
        """
        try:
            logger.info(f"Connecting to MCP server: {server_script_path}")
            
            # Server configuration
            server_params = StdioServerParameters(
                command="python",
                args=[server_script_path],
            )

            # Connect to the server
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )

            # Initialize the connection
            await self.session.initialize()

            # List available tools
            tools_result = await self.session.list_tools()
            logger.info("Connected to MCP server successfully")
            logger.info("Available tools:")
            for tool in tools_result.tools:
                logger.info(f"  - {tool.name}: {tool.description}")
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            raise

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Get available tools from the MCP server in Azure OpenAI format.

        Returns:
            A list of tools in Azure OpenAI function calling format.
            
        Raises:
            Exception: If tool retrieval fails
        """
        try:
            tools_result = await self.session.list_tools()
            openai_tools = [
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
            
            logger.info(f"Retrieved {len(openai_tools)} tools from MCP server")
            return openai_tools
            
        except Exception as e:
            logger.error(f"Failed to retrieve MCP tools: {str(e)}")
            raise

    async def process_query(self, query: str, max_retries: int = 3) -> str:
        """
        Process a query using Azure OpenAI and available MCP tools with retry logic.

        Args:
            query: The user query.
            max_retries: Maximum number of retry attempts for transient failures.

        Returns:
            The response from Azure OpenAI.
            
        Raises:
            Exception: If query processing fails after all retries
        """
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Processing query (attempt {attempt + 1}/{max_retries + 1}): {query[:50]}...")
                
                # Get available tools
                tools = await self.get_mcp_tools()

                # Initial Azure OpenAI API call
                response = await self.azure_openai_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": query}],
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=1500
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
                        try:
                            logger.info(f"Executing tool: {tool_call.function.name}")
                            
                            # Execute tool call
                            result = await self.session.call_tool(
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
                            
                            logger.info(f"Tool {tool_call.function.name} executed successfully")
                            
                        except Exception as e:
                            logger.error(f"Tool execution failed for {tool_call.function.name}: {str(e)}")
                            # Add error message to conversation
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": f"Error executing tool: {str(e)}",
                                }
                            )

                    # Get final response from Azure OpenAI with tool results
                    final_response = await self.azure_openai_client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages,
                        tools=tools,
                        tool_choice="none",  # Don't allow more tool calls
                        temperature=0.7,
                        max_tokens=1500
                    )

                    result = final_response.choices[0].message.content
                    logger.info("Query processed successfully with tool calls")
                    return result

                # No tool calls, just return the direct response
                result = assistant_message.content
                logger.info("Query processed successfully without tool calls")
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries:
                    logger.error(f"All {max_retries + 1} attempts failed. Final error: {str(e)}")
                    raise
                else:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)

    async def cleanup(self):
        """
        Clean up resources and close connections.
        """
        try:
            await self.exit_stack.aclose()
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


async def main():
    """
    Main entry point demonstrating Azure OpenAI MCP integration.
    """
    client = None
    try:
        logger.info("Starting Azure OpenAI MCP integration demo")
        
        # Initialize client
        client = AzureOpenAIMCPClient()
        
        # Connect to MCP server
        await client.connect_to_server("server.py")

        # Example queries
        queries = [
            "What is our company's vacation policy?",
            "How do I request a new software license?",
            "What are the steps for reporting a security incident?",
            "Can you summarize all our remote work policies?"
        ]

        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")

            try:
                response = await client.process_query(query)
                print(f"\nResponse: {response}")
            except Exception as e:
                print(f"\nError processing query: {str(e)}")
                logger.error(f"Query failed: {query} - {str(e)}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"\nApplication error: {str(e)}")
        print("\nPlease check:")
        print("1. Environment variables are properly set")
        print("2. Azure OpenAI service is accessible")
        print("3. MCP server dependencies are installed")
        
    finally:
        # Clean up resources
        if client:
            await client.cleanup()
        logger.info("Application completed")


if __name__ == "__main__":
    asyncio.run(main())
