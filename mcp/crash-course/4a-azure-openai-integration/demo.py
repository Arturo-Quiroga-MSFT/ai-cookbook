"""
MCP Server Demo - Test MCP Server Functionality

This script demonstrates the MCP server functionality without requiring
Azure OpenAI credentials. It directly tests the MCP server tools to
show how they work.
"""

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mcp_server():
    """Test the MCP server directly without Azure OpenAI."""
    exit_stack = AsyncExitStack()
    
    try:
        logger.info("Testing MCP server functionality")
        
        # Server configuration
        server_params = StdioServerParameters(
            command="python",
            args=["server.py"],
        )

        # Connect to the server
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        session = await exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )

        # Initialize the connection
        await session.initialize()
        logger.info("✅ MCP server connection established")

        # List available tools
        tools_result = await session.list_tools()
        logger.info("📋 Available tools:")
        for tool in tools_result.tools:
            logger.info(f"  - {tool.name}: {tool.description}")

        # Test the get_knowledge_base tool
        logger.info("\n🔧 Testing get_knowledge_base tool...")
        result = await session.call_tool("get_knowledge_base", {})
        
        print("\n" + "="*60)
        print("KNOWLEDGE BASE TOOL TEST RESULT")
        print("="*60)
        print(result.content[0].text)
        print("="*60)
        
        logger.info("✅ MCP server test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ MCP server test failed: {str(e)}")
        raise
    finally:
        await exit_stack.aclose()

async def main():
    """Main entry point for the MCP server demo."""
    print("MCP Server Functionality Demo")
    print("=" * 40)
    print("This demo tests the MCP server without requiring Azure OpenAI credentials.")
    print("It shows how the knowledge base tool works independently.\n")
    
    try:
        await test_mcp_server()
        
        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Set up your Azure OpenAI environment variables in .env")
        print("2. Run client-simple.py or client.py to test with Azure OpenAI")
        print("3. Customize the knowledge base in data/kb.json")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure 'mcp' package is installed: pip install mcp")
        print("2. Check that server.py is in the current directory")
        print("3. Verify that data/kb.json exists and is valid JSON")

if __name__ == "__main__":
    asyncio.run(main())
