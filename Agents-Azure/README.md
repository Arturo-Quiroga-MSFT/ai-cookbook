# The 7 Foundational Building Blocks of AI Agents - Azure OpenAI Edition

## Overview

This folder contains modified versions of the foundational building blocks that work with Azure OpenAI models instead of the standard OpenAI API. All examples have been adapted to use Azure OpenAI endpoints and authentication.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   - Copy `.env.example` to `.env`
   - Fill in your Azure OpenAI configuration:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` with your Azure OpenAI details:
     - `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI resource endpoint
     - `AZURE_OPENAI_API_KEY`: Your API key (for development)
     - `AZURE_OPENAI_API_VERSION`: API version (recommended: 2024-12-01-preview)
     - `AZURE_OPENAI_GPT4_DEPLOYMENT`: Your GPT-4 deployment name
     - `AZURE_OPENAI_GPT4_MINI_DEPLOYMENT`: Your GPT-4 mini deployment name

3. **Authentication Options:**
   - **Development**: Use API key authentication (configured in `.env`)
   - **Production**: Use Managed Identity (recommended for Azure-hosted applications)
   - **CI/CD**: Use Service Principal authentication

## Key Changes from Original Building Blocks

1. **Azure OpenAI Client**: Uses `AzureOpenAI` client instead of `OpenAI`
2. **Environment Configuration**: All settings loaded from `.env` file
3. **Deployment Names**: Uses Azure OpenAI deployment names instead of model names
4. **Authentication**: Supports both API key and Managed Identity authentication
5. **Error Handling**: Enhanced with Azure-specific error handling
6. **Security**: Follows Azure security best practices

## Files

### Configuration
- **`.env.example`**: Template for environment variables
- **`requirements.txt`**: Python dependencies including Azure-specific packages

### Building Blocks
- **`1-intelligence.py`**: Azure OpenAI intelligence implementation
- **`2-memory.py`**: Memory management with Azure OpenAI
- **`3-tools.py`**: Tool integration with Azure OpenAI function calling
- **`4-validation.py`**: Structured output validation using Azure OpenAI
- **`5-control.py`**: Control flow and decision-making with Azure OpenAI
- **`6-recovery.py`**: Error handling and recovery patterns
- **`7-feedback.py`**: Human-in-the-loop workflows

## Security Best Practices

- **Never commit `.env` files**: Add `.env` to your `.gitignore`
- **Use Managed Identity**: In production, use Azure Managed Identity instead of API keys
- **Least Privilege**: Grant minimal required permissions to your Azure OpenAI resource
- **Monitor Usage**: Enable logging and monitoring for your Azure OpenAI resource
- **Rotate Keys**: Regularly rotate API keys if using key-based authentication

## Azure OpenAI Specific Features

- **Content Filtering**: Azure OpenAI includes built-in content filtering
- **Regional Deployment**: Models are deployed in specific Azure regions
- **Enterprise Features**: Enhanced security, compliance, and governance features
- **Private Endpoints**: Support for private network access

## Running the Examples

Each file can be run independently:

```bash
python 1-intelligence.py
python 2-memory.py
python 3-tools.py
python 4-validation.py
python 5-control.py
python 6-recovery.py
python 7-feedback.py
```

## Troubleshooting

- **Authentication Issues**: Verify your endpoint URL and API key
- **Deployment Not Found**: Ensure deployment names match your Azure OpenAI resource
- **Rate Limiting**: Azure OpenAI has different rate limits than OpenAI API
- **Regional Availability**: Some models may not be available in all Azure regions

## Original Documentation

For more details on the foundational concepts, see the [original building blocks documentation](../agents/building-blocks/README.md).
