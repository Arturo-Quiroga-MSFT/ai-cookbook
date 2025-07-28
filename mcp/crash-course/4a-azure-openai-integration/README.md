# Azure OpenAI Integration with MCP

This section demonstrates how to integrate the Model Context Protocol (MCP) with Azure OpenAI to create a system where Azure OpenAI can access and use tools provided by your MCP server.

## Overview

This example shows how to:

1. Create an MCP server that exposes a knowledge base tool
2. Connect Azure OpenAI to this MCP server
3. Allow Azure OpenAI to dynamically use the tools when responding to user queries
4. Support both API key and Managed Identity authentication methods

## Key Differences from OpenAI Integration

This Azure OpenAI version provides:

- **Azure OpenAI Client Integration**: Uses `AsyncAzureOpenAI` instead of `AsyncOpenAI`
- **Flexible Authentication**: Supports both API key and Azure Managed Identity
- **Azure-Specific Configuration**: Endpoint and deployment name configuration
- **Enhanced Error Handling**: Azure-specific error handling and retry logic
- **Production Best Practices**: Comprehensive logging, monitoring, and security

## Authentication Methods

### 1. API Key Authentication (Recommended for Development)
Set these environment variables:
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_KEY=your-api-key
```

### 2. Managed Identity (Recommended for Production)
Set these environment variables:
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
# No API key needed - uses DefaultAzureCredential
```

For Managed Identity, ensure your application has the "Cognitive Services OpenAI User" role assigned.

## Connection Methods

This example uses the **stdio transport** for communication between the client and server, which means:

- The client and server run in the same process
- The client directly launches the server as a subprocess
- No separate server process is needed

If you want to split your client and server into separate applications (e.g., running the server on a different machine), you'll need to use the **SSE (Server-Sent Events) transport** instead. For details on setting up an SSE connection, see the [Simple Server Setup](../3-simple-server-setup) section.

### Data Flow Explanation

1. **User Query**: The user sends a query to the system (e.g., "What is our company's vacation policy?")
2. **Azure OpenAI API**: Azure OpenAI receives the query and available tools from the MCP server
3. **Tool Selection**: Azure OpenAI decides which tools to use based on the query
4. **MCP Client**: The client receives Azure OpenAI's tool call request and forwards it to the MCP server
5. **MCP Server**: The server executes the requested tool (e.g., retrieving knowledge base data)
6. **Response Flow**: The tool result flows back through the MCP client to Azure OpenAI
7. **Final Response**: Azure OpenAI generates a final response incorporating the tool data

## How Azure OpenAI Executes Tools

Azure OpenAI's function calling mechanism works with MCP tools through these steps:

1. **Tool Registration**: The MCP client converts MCP tools to Azure OpenAI's function format
2. **Tool Choice**: Azure OpenAI decides which tools to use based on the user query
3. **Tool Execution**: The MCP client executes the selected tools and returns results
4. **Context Integration**: Azure OpenAI incorporates the tool results into its response

## The Role of MCP

MCP serves as a standardized bridge between AI models and your backend systems:

- **Standardization**: MCP provides a consistent interface for AI models to interact with tools
- **Abstraction**: MCP abstracts away the complexity of your backend systems
- **Security**: MCP allows you to control exactly what tools and data are exposed to AI models
- **Flexibility**: You can change your backend implementation without changing the AI integration

## Implementation Details

### Azure Utilities (`azure_utils.py`)

Provides shared utilities for Azure OpenAI integration:
- Client initialization with authentication support
- Environment validation
- Configuration management
- Error handling utilities

### Server (`server.py`)

The MCP server exposes a `get_knowledge_base` tool that retrieves Q&A pairs from a JSON file. The server is identical to the OpenAI version since it doesn't directly interact with the AI model.

### Advanced Client (`client.py`)

The advanced client provides:
- Comprehensive error handling and retry logic
- Support for multiple authentication methods
- Detailed logging and monitoring
- Clean resource management
- Production-ready implementation

### Simple Client (`client-simple.py`)

The simple client provides:
- Easy-to-understand implementation
- Basic error handling
- Good starting point for customization
- Minimal dependencies

### Knowledge Base (`data/kb.json`)

Contains Q&A pairs about company policies that can be queried through the MCP server.

## Environment Configuration

Create a `.env` file in the root directory with your Azure OpenAI configuration:

```bash
# Required for all authentication methods
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name

# Required for API key authentication
AZURE_OPENAI_API_KEY=your-api-key

# Optional for Managed Identity (if using specific client ID)
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_TENANT_ID=your-tenant-id
```

## Running the Examples

### Prerequisites

1. Install the required dependencies:
   ```bash
   pip install openai azure-identity python-dotenv mcp nest-asyncio
   ```

2. Set up your Azure OpenAI configuration in the `.env` file

3. Ensure your Azure OpenAI deployment is accessible

### Run the Advanced Client

```bash
python client.py
```

### Run the Simple Client

```bash
python client-simple.py
```

Note: With the stdio transport used in these examples, you don't need to run the server separately as the client will automatically start it.

## Error Handling and Troubleshooting

The clients include comprehensive error handling for common issues:

- **Environment validation**: Checks for required variables before starting
- **Authentication errors**: Clear messages for authentication failures
- **Network issues**: Retry logic with exponential backoff
- **Tool execution errors**: Graceful handling of tool failures
- **Resource cleanup**: Proper cleanup of connections and resources

### Common Issues

1. **Environment Variables**: Ensure all required variables are set in your `.env` file
2. **Authentication**: Verify your API key or Managed Identity configuration
3. **Network Access**: Ensure your Azure OpenAI endpoint is accessible
4. **Deployment Name**: Verify the deployment name matches your Azure OpenAI resource

## Security Best Practices

- **Never hardcode credentials**: Always use environment variables or Managed Identity
- **Use Managed Identity in production**: More secure than API keys
- **Limit tool access**: Only expose necessary tools through MCP
- **Monitor usage**: Enable logging and monitoring for production deployments
- **Validate inputs**: Ensure proper validation of user inputs and tool responses

## Next Steps

1. **Customize the knowledge base**: Add your own company data to `data/kb.json`
2. **Add more tools**: Extend the MCP server with additional tools
3. **Implement caching**: Add caching for frequently accessed data
4. **Add authentication**: Implement user authentication for production use
5. **Scale the deployment**: Consider containerization and orchestration
