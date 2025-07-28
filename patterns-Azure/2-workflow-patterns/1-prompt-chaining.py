"""
Prompt Chaining with Azure OpenAI.
Demonstrates breaking down complex AI tasks into a sequence of smaller, focused steps.
Each step processes the output from the previous step for better control and reliability.
"""

import sys
import os
import logging
from typing import Optional
from datetime import datetime

# Add parent directory to path for azure_utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure_utils import get_azure_openai_client, get_deployment_name, validate_environment
from pydantic import BaseModel, Field

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------
# Step 1: Define the data models for each stage
# --------------------------------------------------------------

class EventExtraction(BaseModel):
    """First Azure OpenAI call: Extract basic event information"""
    
    description: str = Field(description="Raw description of the event")
    is_calendar_event: bool = Field(
        description="Whether this text describes a calendar event"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")


class EventDetails(BaseModel):
    """Second Azure OpenAI call: Parse specific event details"""
    
    name: str = Field(description="Name of the event")
    date: str = Field(
        description="Date and time of the event. Use ISO 8601 to format this value."
    )
    duration_minutes: int = Field(description="Expected duration in minutes")
    participants: list[str] = Field(description="List of participants")


class EventConfirmation(BaseModel):
    """Third Azure OpenAI call: Generate confirmation message"""
    
    confirmation_message: str = Field(
        description="Natural language confirmation message"
    )
    calendar_link: Optional[str] = Field(
        description="Generated calendar link if applicable"
    )


# --------------------------------------------------------------
# Step 2: Define the chaining functions
# --------------------------------------------------------------

def extract_event_info(client, deployment_name: str, user_input: str) -> EventExtraction:
    """First Azure OpenAI call to determine if input is a calendar event"""
    logger.info("Starting event extraction analysis")
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    completion = client.beta.chat.completions.parse(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Analyze if the text describes a calendar event.",
            },
            {"role": "user", "content": user_input},
        ],
        response_format=EventExtraction,
        temperature=0.1  # Lower temperature for more consistent analysis
    )
    
    result = completion.choices[0].message.parsed
    logger.info(
        f"Extraction complete - Is calendar event: {result.is_calendar_event}, Confidence: {result.confidence_score:.2f}"
    )
    return result


def parse_event_details(client, deployment_name: str, description: str) -> EventDetails:
    """Second Azure OpenAI call to extract specific event details"""
    logger.info("Starting event details parsing")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    completion = client.beta.chat.completions.parse(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Extract detailed event information. When dates reference 'next Tuesday' or similar relative dates, use this current date as reference.",
            },
            {"role": "user", "content": description},
        ],
        response_format=EventDetails,
        temperature=0.1
    )
    
    result = completion.choices[0].message.parsed
    logger.info(
        f"Parsed event details - Name: {result.name}, Date: {result.date}, Duration: {result.duration_minutes}min"
    )
    logger.debug(f"Participants: {', '.join(result.participants)}")
    return result


def generate_confirmation(client, deployment_name: str, event_details: EventDetails) -> EventConfirmation:
    """Third Azure OpenAI call to generate a confirmation message"""
    logger.info("Generating confirmation message")

    completion = client.beta.chat.completions.parse(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": "Generate a natural confirmation message for the event. Sign off with your name; Azure Assistant",
            },
            {"role": "user", "content": str(event_details.model_dump())},
        ],
        response_format=EventConfirmation,
        temperature=0.3  # Slightly higher temperature for more natural language
    )
    
    result = completion.choices[0].message.parsed
    logger.info("Confirmation message generated successfully")
    return result


# --------------------------------------------------------------
# Step 3: Chain the functions together
# --------------------------------------------------------------

def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    """
    Main function implementing the prompt chain with gate check using Azure OpenAI.
    
    Args:
        user_input: Raw user input to process
        
    Returns:
        EventConfirmation or None if not a valid calendar event
    """
    logger.info("Processing calendar request with Azure OpenAI")
    logger.debug(f"Raw input: {user_input}")

    try:
        # Initialize Azure OpenAI client
        client = get_azure_openai_client()
        deployment_name = get_deployment_name("gpt4")

        # First Azure OpenAI call: Extract basic info
        initial_extraction = extract_event_info(client, deployment_name, user_input)

        # Gate check: Verify if it's a calendar event with sufficient confidence
        if (
            not initial_extraction.is_calendar_event
            or initial_extraction.confidence_score < 0.7
        ):
            logger.warning(
                f"Gate check failed - is_calendar_event: {initial_extraction.is_calendar_event}, confidence: {initial_extraction.confidence_score:.2f}"
            )
            return None

        logger.info("Gate check passed, proceeding with event processing")

        # Second Azure OpenAI call: Get detailed event information
        event_details = parse_event_details(client, deployment_name, initial_extraction.description)

        # Third Azure OpenAI call: Generate confirmation
        confirmation = generate_confirmation(client, deployment_name, event_details)

        logger.info("Calendar request processing completed successfully")
        return confirmation

    except Exception as e:
        logger.error(f"Error in calendar request processing: {e}")
        return None


def demonstrate_prompt_chaining():
    """Demonstrate the prompt chaining pattern with Azure OpenAI."""
    
    print("Azure OpenAI Prompt Chaining Demo")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "input": "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap.",
            "description": "Valid calendar event request"
        },
        {
            "input": "Can you send an email to Alice and Bob to discuss the project roadmap?",
            "description": "Non-calendar request (should fail gate check)"
        },
        {
            "input": "Schedule a quick 30-minute standup tomorrow at 9am with the development team.",
            "description": "Another valid calendar event"
        },
        {
            "input": "What's the weather like today?",
            "description": "Unrelated question (should fail gate check)"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Input: {test_case['input']}")
        print("-" * 40)
        
        result = process_calendar_request(test_case['input'])
        
        if result:
            print("✅ Successfully processed as calendar event")
            print(f"Confirmation: {result.confirmation_message}")
            if result.calendar_link:
                print(f"Calendar Link: {result.calendar_link}")
        else:
            print("❌ Not processed as calendar event (failed gate check)")
        
        print()


def main():
    """Main demonstration of prompt chaining with Azure OpenAI."""
    
    # Validate environment configuration
    if not validate_environment():
        print("Environment validation failed. Please check your .env configuration.")
        return
    
    demonstrate_prompt_chaining()


if __name__ == "__main__":
    main()
