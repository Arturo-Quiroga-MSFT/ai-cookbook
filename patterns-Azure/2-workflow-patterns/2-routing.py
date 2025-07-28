"""
Routing with Azure OpenAI.
Demonstrates directing different types of requests to specialized handlers.
This allows for optimized processing of distinct request types while maintaining clean separation of concerns.
"""

import sys
import os
import logging
from typing import Optional, Literal

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
# Step 1: Define the data models for routing and responses
# --------------------------------------------------------------

class CalendarRequestType(BaseModel):
    """Router Azure OpenAI call: Determine the type of calendar request"""
    
    request_type: Literal["new_event", "modify_event", "other"] = Field(
        description="Type of calendar request being made"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    description: str = Field(description="Cleaned description of the request")


class NewEventDetails(BaseModel):
    """Details for creating a new event"""
    
    name: str = Field(description="Name of the event")
    date: str = Field(description="Date and time of the event (ISO 8601)")
    duration_minutes: int = Field(description="Duration in minutes")
    participants: list[str] = Field(description="List of participants")


class Change(BaseModel):
    """Details for changing an existing event"""
    
    field: str = Field(description="Field to change")
    new_value: str = Field(description="New value for the field")


class ModifyEventDetails(BaseModel):
    """Details for modifying an existing event"""
    
    event_identifier: str = Field(
        description="Description to identify the existing event"
    )
    changes: list[Change] = Field(description="List of changes to make")
    participants_to_add: list[str] = Field(description="New participants to add")
    participants_to_remove: list[str] = Field(description="Participants to remove")


class CalendarResponse(BaseModel):
    """Final response format"""
    
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="User-friendly response message")
    calendar_link: Optional[str] = Field(description="Calendar link if applicable")


# --------------------------------------------------------------
# Step 2: Define the routing and processing functions
# --------------------------------------------------------------

def route_calendar_request(client, deployment_name: str, user_input: str) -> CalendarRequestType:
    """Router Azure OpenAI call to determine the type of calendar request"""
    logger.info("Routing calendar request with Azure OpenAI")

    completion = client.beta.chat.completions.parse(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": "Determine if this is a request to create a new calendar event or modify an existing one. Analyze the intent carefully.",
            },
            {"role": "user", "content": user_input},
        ],
        response_format=CalendarRequestType,
        temperature=0.1  # Lower temperature for consistent routing decisions
    )
    
    result = completion.choices[0].message.parsed
    logger.info(
        f"Request routed as: {result.request_type} with confidence: {result.confidence_score:.2f}"
    )
    return result


def handle_new_event(client, deployment_name: str, description: str) -> CalendarResponse:
    """Process a new event request using Azure OpenAI"""
    logger.info("Processing new event request")

    # Get event details
    completion = client.beta.chat.completions.parse(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": "Extract details for creating a new calendar event. Use ISO 8601 format for dates.",
            },
            {"role": "user", "content": description},
        ],
        response_format=NewEventDetails,
        temperature=0.1
    )
    
    details = completion.choices[0].message.parsed
    logger.info(f"New event details extracted: {details.name}")

    # Generate response
    return CalendarResponse(
        success=True,
        message=f"✅ Created new event '{details.name}' for {details.date} with {', '.join(details.participants)} (Duration: {details.duration_minutes} minutes)",
        calendar_link=f"calendar://new?event={details.name.replace(' ', '%20')}"
    )


def handle_modify_event(client, deployment_name: str, description: str) -> CalendarResponse:
    """Process an event modification request using Azure OpenAI"""
    logger.info("Processing event modification request")

    # Get modification details
    completion = client.beta.chat.completions.parse(
        model=deployment_name,
        messages=[
            {
                "role": "system",
                "content": "Extract details for modifying an existing calendar event. Identify what changes are being requested.",
            },
            {"role": "user", "content": description},
        ],
        response_format=ModifyEventDetails,
        temperature=0.1
    )
    
    details = completion.choices[0].message.parsed
    logger.info(f"Modify event details extracted for: {details.event_identifier}")

    # Build change summary
    change_summary = []
    for change in details.changes:
        change_summary.append(f"{change.field} → {change.new_value}")
    
    if details.participants_to_add:
        change_summary.append(f"Added participants: {', '.join(details.participants_to_add)}")
    
    if details.participants_to_remove:
        change_summary.append(f"Removed participants: {', '.join(details.participants_to_remove)}")

    # Generate response
    changes_text = "; ".join(change_summary) if change_summary else "general modifications"
    
    return CalendarResponse(
        success=True,
        message=f"🔄 Modified event '{details.event_identifier}' with changes: {changes_text}",
        calendar_link=f"calendar://modify?event={details.event_identifier.replace(' ', '%20')}"
    )


def process_calendar_request(user_input: str) -> Optional[CalendarResponse]:
    """
    Main function implementing the routing workflow with Azure OpenAI.
    
    Args:
        user_input: Raw user input to process
        
    Returns:
        CalendarResponse or None if request cannot be processed
    """
    logger.info("Processing calendar request with Azure OpenAI routing")

    try:
        # Initialize Azure OpenAI client
        client = get_azure_openai_client()
        deployment_name = get_deployment_name("gpt4")

        # Route the request
        route_result = route_calendar_request(client, deployment_name, user_input)

        # Check confidence threshold
        if route_result.confidence_score < 0.7:
            logger.warning(f"Low confidence score: {route_result.confidence_score:.2f}")
            return CalendarResponse(
                success=False,
                message="❓ I'm not confident this is a calendar request. Could you please clarify?",
                calendar_link=None
            )

        # Route to appropriate handler
        if route_result.request_type == "new_event":
            return handle_new_event(client, deployment_name, route_result.description)
        elif route_result.request_type == "modify_event":
            return handle_modify_event(client, deployment_name, route_result.description)
        else:
            logger.warning("Request type not supported for calendar operations")
            return CalendarResponse(
                success=False,
                message="❌ This doesn't appear to be a calendar-related request.",
                calendar_link=None
            )

    except Exception as e:
        logger.error(f"Error in calendar request processing: {e}")
        return CalendarResponse(
            success=False,
            message="⚠️ An error occurred while processing your calendar request.",
            calendar_link=None
        )


def demonstrate_routing():
    """Demonstrate the routing pattern with Azure OpenAI."""
    
    print("Azure OpenAI Routing Demo")
    print("=" * 50)
    
    # Test cases for different routing scenarios
    test_cases = [
        {
            "input": "Let's schedule a team meeting next Tuesday at 2pm with Alice and Bob",
            "description": "New event creation request",
            "expected_route": "new_event"
        },
        {
            "input": "Can you move the team meeting with Alice and Bob to Wednesday at 3pm instead?",
            "description": "Event modification request",
            "expected_route": "modify_event"
        },
        {
            "input": "Schedule a 1-hour standup tomorrow at 9am with the dev team",
            "description": "Another new event request",
            "expected_route": "new_event"
        },
        {
            "input": "Change the client call from 2pm to 4pm and add Sarah to the invite list",
            "description": "Complex modification request",
            "expected_route": "modify_event"
        },
        {
            "input": "What's the weather like today?",
            "description": "Non-calendar request (should fail)",
            "expected_route": "other"
        },
        {
            "input": "Send an email to the team about the project update",
            "description": "Email request (should fail)",
            "expected_route": "other"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Input: {test_case['input']}")
        print(f"Expected Route: {test_case['expected_route']}")
        print("-" * 40)
        
        result = process_calendar_request(test_case['input'])
        
        if result:
            status_icon = "✅" if result.success else "❌"
            print(f"{status_icon} {result.message}")
            if result.calendar_link:
                print(f"📅 Calendar Link: {result.calendar_link}")
        else:
            print("❌ No response generated")
        
        print()


def main():
    """Main demonstration of routing with Azure OpenAI."""
    
    # Validate environment configuration
    if not validate_environment():
        print("Environment validation failed. Please check your .env configuration.")
        return
    
    demonstrate_routing()


if __name__ == "__main__":
    main()
