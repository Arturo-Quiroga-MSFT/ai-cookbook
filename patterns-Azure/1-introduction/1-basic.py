"""
Basic Azure OpenAI example.
Demonstrates simple chat completion using Azure OpenAI service.
"""

import sys
import os

# Add parent directory to path for azure_utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure_utils import get_azure_openai_client, get_deployment_name, validate_environment

def main():
    """Basic Azure OpenAI chat completion example."""
    
    # Validate environment configuration
    if not validate_environment():
        print("Environment validation failed. Please check your .env configuration.")
        return
    
    try:
        # Initialize Azure OpenAI client
        client = get_azure_openai_client()
        deployment_name = get_deployment_name("gpt4")
        
        # Make chat completion request
        completion = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You're a helpful assistant."},
                {
                    "role": "user",
                    "content": "Write a limerick about the Python programming language.",
                },
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        response = completion.choices[0].message.content
        print("Azure OpenAI Response:")
        print("=" * 50)
        print(response)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your Azure OpenAI configuration is correct in .env file")


if __name__ == "__main__":
    main()
