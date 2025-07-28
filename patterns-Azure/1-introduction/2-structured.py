"""
Structured output with Azure OpenAI.
Demonstrates using Pydantic models for structured responses from Azure OpenAI.
"""

import sys
import os

# Add parent directory to path for azure_utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure_utils import get_azure_openai_client, get_deployment_name, validate_environment
from pydantic import BaseModel
from typing import List


# --------------------------------------------------------------
# Step 1: Define the response format in a Pydantic model
# --------------------------------------------------------------

class CalendarEvent(BaseModel):
    """Calendar event data model."""
    name: str
    date: str
    participants: List[str]


def main():
    """Structured output example with Azure OpenAI."""
    
    # Validate environment configuration
    if not validate_environment():
        print("Environment validation failed. Please check your .env configuration.")
        return
    
    try:
        # Initialize Azure OpenAI client
        client = get_azure_openai_client()
        deployment_name = get_deployment_name("gpt4")
        
        # --------------------------------------------------------------
        # Step 2: Call the model with structured output
        # --------------------------------------------------------------
        
        completion = client.beta.chat.completions.parse(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "Extract the event information."},
                {
                    "role": "user",
                    "content": "Alice and Bob are going to a science fair on Friday.",
                },
            ],
            response_format=CalendarEvent,
            temperature=0.0  # Lower temperature for more consistent structured output
        )
        
        # --------------------------------------------------------------
        # Step 3: Parse the response
        # --------------------------------------------------------------
        
        event = completion.choices[0].message.parsed
        
        print("Extracted Calendar Event:")
        print("=" * 30)
        print(f"Event Name: {event.name}")
        print(f"Date: {event.date}")
        print(f"Participants: {', '.join(event.participants)}")
        
        # Demonstrate accessing individual fields
        print(f"\nIndividual field access:")
        print(f"- Name: {event.name}")
        print(f"- Date: {event.date}")
        print(f"- Participants: {event.participants}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your Azure OpenAI configuration is correct in .env file")


if __name__ == "__main__":
    main()
