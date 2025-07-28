"""
MCP Server for Knowledge Base Tool with Azure OpenAI Integration.

This server exposes a knowledge base tool that can be used by Azure OpenAI models
through the Model Context Protocol (MCP). The server runs independently of the
AI model and provides secure access to company knowledge base data.
"""

import os
import json
import logging
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create an MCP server
mcp = FastMCP(
    name="Knowledge Base",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8050,  # only used for SSE transport (set this to any port)
)


@mcp.tool()
def get_knowledge_base() -> str:
    """Retrieve the entire knowledge base as a formatted string.

    This tool provides access to company policies and procedures stored in a JSON file.
    The knowledge base contains Q&A pairs covering various company topics including
    vacation policies, software licensing, remote work guidelines, expense reporting,
    and security incident procedures.

    Returns:
        A formatted string containing all Q&A pairs from the knowledge base.
    """
    try:
        # Get the absolute path to the knowledge base file
        kb_path = os.path.join(os.path.dirname(__file__), "data", "kb.json")
        logger.info(f"Attempting to load knowledge base from: {kb_path}")
        
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_data = json.load(f)

        # Format the knowledge base as a string
        kb_text = "Here is the retrieved knowledge base:\n\n"

        if isinstance(kb_data, list):
            for i, item in enumerate(kb_data, 1):
                if isinstance(item, dict):
                    question = item.get("question", "Unknown question")
                    answer = item.get("answer", "Unknown answer")
                else:
                    question = f"Item {i}"
                    answer = str(item)

                kb_text += f"Q{i}: {question}\n"
                kb_text += f"A{i}: {answer}\n\n"
        else:
            kb_text += f"Knowledge base content: {json.dumps(kb_data, indent=2)}\n\n"

        logger.info(f"Successfully loaded knowledge base with {len(kb_data) if isinstance(kb_data, list) else 1} entries")
        return kb_text
        
    except FileNotFoundError:
        error_msg = "Error: Knowledge base file not found"
        logger.error(f"{error_msg}. Expected path: {kb_path}")
        return error_msg
    except json.JSONDecodeError as e:
        error_msg = f"Error: Invalid JSON in knowledge base file - {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(f"Unexpected error loading knowledge base: {error_msg}")
        return error_msg


# Run the server
if __name__ == "__main__":
    logger.info("Starting MCP Knowledge Base Server for Azure OpenAI integration")
    logger.info("Available tools:")
    logger.info("  - get_knowledge_base: Retrieve company knowledge base Q&A pairs")
    
    # Run with stdio transport for direct client-server communication
    mcp.run(transport="stdio")
