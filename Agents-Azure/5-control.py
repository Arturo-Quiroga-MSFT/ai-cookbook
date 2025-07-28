"""
Control: Provides deterministic decision-making and process flow control using Azure OpenAI.
This component handles if/then logic, routing based on conditions, and process orchestration for predictable behavior.

Azure OpenAI Documentation: https://docs.microsoft.com/en-us/azure/cognitive-services/openai/
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
import json

# Load environment variables
load_dotenv()


def get_azure_openai_client():
    """Initialize Azure OpenAI client with proper authentication."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
    
    if api_key:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    else:
        credential = DefaultAzureCredential()
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=credential,
            api_version=api_version,
        )
    
    return client


class IntentClassification(BaseModel):
    """
    Schema for intent classification results.
    """
    intent: Literal["question", "request", "complaint", "greeting", "goodbye"] = Field(
        description="The classified intent of the user input"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, 
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        description="Explanation for the classification decision"
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Key entities extracted from the input"
    )


class RoutingDecision(BaseModel):
    """
    Schema for routing decisions.
    """
    department: Literal["sales", "support", "billing", "technical", "general"] = Field(
        description="Department to route the request to"
    )
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        description="Priority level of the request"
    )
    requires_human: bool = Field(
        description="Whether human intervention is required"
    )
    estimated_resolution_time: str = Field(
        description="Estimated time to resolve (e.g., '5 minutes', '2 hours', '1 day')"
    )


def classify_intent(user_input: str) -> IntentClassification:
    """
    Classify user input into predefined intent categories.
    
    Args:
        user_input: The user's input message
        
    Returns:
        Intent classification with confidence and reasoning
    """
    client = get_azure_openai_client()
    deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
    
    system_prompt = """
    Classify the user input into one of these intents: question, request, complaint, greeting, goodbye.
    
    Definitions:
    - question: User is asking for information or explanation
    - request: User wants something to be done or executed
    - complaint: User is expressing dissatisfaction or reporting a problem
    - greeting: User is initiating conversation or saying hello
    - goodbye: User is ending conversation or saying farewell
    
    Return JSON in exactly this format:
    {
        "intent": "one of: question|request|complaint|greeting|goodbye",
        "confidence": 0.95,
        "reasoning": "explanation for the classification",
        "entities": ["list", "of", "key", "words", "or", "phrases"]
    }
    
    Important: entities must be an array of strings, not an object.
    """
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        json_data = json.loads(response_text)
        return IntentClassification(**json_data)
        
    except Exception as e:
        print(f"Intent classification error: {e}")
        # Fallback classification
        return IntentClassification(
            intent="question",
            confidence=0.0,
            reasoning=f"Error in classification: {str(e)}",
            entities=[]
        )


def route_request(user_input: str, intent: IntentClassification) -> RoutingDecision:
    """
    Route requests to appropriate departments based on content and intent.
    
    Args:
        user_input: Original user input
        intent: Classified intent
        
    Returns:
        Routing decision with department and priority
    """
    client = get_azure_openai_client()
    deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
    
    system_prompt = f"""
    Route this request to the appropriate department based on the content and intent.
    
    User intent: {intent.intent}
    User input: {user_input}
    
    Departments:
    - sales: Product inquiries, pricing, demos, new business
    - support: Product issues, how-to questions, troubleshooting
    - billing: Payment issues, invoices, subscription changes
    - technical: API problems, integration issues, advanced technical questions
    - general: Greetings, general inquiries, unclear requests
    
    Priority levels:
    - urgent: System down, security issues, payment failures
    - high: Product not working, blocking issues, angry customers
    - medium: General questions, feature requests, minor issues
    - low: Documentation requests, general inquiries
    
    Return JSON in exactly this format:
    {{
        "department": "one of: sales|support|billing|technical|general",
        "priority": "one of: low|medium|high|urgent",
        "requires_human": true,
        "estimated_resolution_time": "5 minutes"
    }}
    
    Important: All fields are required. estimated_resolution_time should be a time estimate like "5 minutes", "2 hours", "1 day".
    """
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Route this: {user_input}"},
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        json_data = json.loads(response_text)
        return RoutingDecision(**json_data)
        
    except Exception as e:
        print(f"Routing error: {e}")
        # Fallback routing
        return RoutingDecision(
            department="general",
            priority="medium",
            requires_human=True,
            estimated_resolution_time="30 minutes"
        )


def route_based_on_intent(user_input: str) -> tuple[str, IntentClassification, RoutingDecision]:
    """
    Complete routing workflow: classify intent and route to appropriate handler.
    
    Args:
        user_input: User's input message
        
    Returns:
        Tuple of (response, intent_classification, routing_decision)
    """
    # Step 1: Classify intent
    classification = classify_intent(user_input)
    intent = classification.intent

    # Step 2: Route based on intent and content
    routing = route_request(user_input, classification)
    
    # Step 3: Handle based on classification
    if intent == "greeting":
        result = handle_greeting(user_input)
    elif intent == "goodbye":
        result = handle_goodbye(user_input)
    elif intent == "question":
        result = answer_question(user_input, routing)
    elif intent == "request":
        result = process_request(user_input, routing)
    elif intent == "complaint":
        result = handle_complaint(user_input, routing)
    else:
        result = "I'm not sure how to help with that. Let me connect you with a human agent."

    return result, classification, routing


def handle_greeting(user_input: str) -> str:
    """Handle greeting messages."""
    return "Hello! How can I assist you today? I can help with questions, requests, or connect you with the right department."


def handle_goodbye(user_input: str) -> str:
    """Handle goodbye messages."""
    return "Thank you for contacting us! Have a great day and feel free to reach out if you need anything else."


def answer_question(question: str, routing: RoutingDecision) -> str:
    """
    Answer questions based on routing decision.
    
    Args:
        question: The user's question
        routing: Routing decision with department info
        
    Returns:
        Appropriate response
    """
    client = get_azure_openai_client()
    deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
    
    department_context = {
        "sales": "You are a sales representative. Focus on how our products can solve their needs.",
        "support": "You are a support agent. Provide helpful troubleshooting and guidance.",
        "billing": "You are a billing specialist. Help with account and payment questions.",
        "technical": "You are a technical expert. Provide detailed technical information.",
        "general": "You are a general assistant. Provide helpful information and guidance."
    }
    
    context = department_context.get(routing.department, department_context["general"])
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question},
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I'm having trouble processing your question right now. Please try again or contact our {routing.department} team directly."


def process_request(request: str, routing: RoutingDecision) -> str:
    """Process user requests based on routing."""
    if routing.requires_human:
        return f"I've routed your request to our {routing.department} team (Priority: {routing.priority}). Expected response time: {routing.estimated_resolution_time}. Someone will get back to you soon!"
    else:
        return f"Processing your request: {request}. This has been categorized as {routing.priority} priority for the {routing.department} team."


def handle_complaint(complaint: str, routing: RoutingDecision) -> str:
    """Handle complaint messages with appropriate escalation."""
    return f"I understand your concern and I'm sorry you're experiencing this issue. I've escalated this to our {routing.department} team with {routing.priority} priority. Expected resolution time: {routing.estimated_resolution_time}. A specialist will contact you shortly to resolve this matter."


class ControlAgent:
    """
    Advanced control agent that manages complex decision trees and workflows.
    """
    
    def __init__(self):
        self.client = get_azure_openai_client()
        self.deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
        self.conversation_history = []
    
    def process_input(self, user_input: str) -> dict:
        """
        Process user input through complete control workflow.
        
        Args:
            user_input: User's message
            
        Returns:
            Complete processing result with all decision points
        """
        # Classify and route
        response, classification, routing = route_based_on_intent(user_input)
        
        # Store in conversation history
        self.conversation_history.append({
            "user_input": user_input,
            "classification": classification.model_dump(),
            "routing": routing.model_dump(),
            "response": response
        })
        
        return {
            "response": response,
            "classification": classification.model_dump(),
            "routing": routing.model_dump(),
            "conversation_turn": len(self.conversation_history)
        }
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation flow."""
        if not self.conversation_history:
            return "No conversation history yet."
        
        intents = [turn["classification"]["intent"] for turn in self.conversation_history]
        departments = [turn["routing"]["department"] for turn in self.conversation_history]
        
        return f"Conversation summary: {len(self.conversation_history)} turns, Intents: {list(set(intents))}, Departments involved: {list(set(departments))}"


if __name__ == "__main__":
    print("=== Control Demo ===\n")
    
    # Test different types of inputs
    test_inputs = [
        "Hello there!",
        "What is machine learning and how does it work?",
        "Please schedule a meeting for tomorrow at 2 PM",
        "I'm unhappy with the service quality, my account is not working",
        "Thanks for your help, goodbye!",
        "My API integration is failing with error 500",
        "How much does your premium plan cost?",
        "I can't log into my account and need help immediately"
    ]

    for i, user_input in enumerate(test_inputs, 1):
        print(f"{i}. Input: '{user_input}'")
        
        try:
            result, classification, routing = route_based_on_intent(user_input)
            
            print(f"   Intent: {classification.intent} (confidence: {classification.confidence:.2f})")
            print(f"   Reasoning: {classification.reasoning}")
            print(f"   Department: {routing.department}, Priority: {routing.priority}")
            print(f"   Requires Human: {routing.requires_human}")
            print(f"   Response: {result[:100]}{'...' if len(result) > 100 else ''}")
            print()
            
        except Exception as e:
            print(f"   Error: {e}")
            print()
    
    # Test the ControlAgent
    print("=== Testing ControlAgent ===\n")
    agent = ControlAgent()
    
    test_conversation = [
        "Hi, I need help with my account",
        "I can't access my dashboard and it's urgent",
        "Thank you for your help"
    ]
    
    for msg in test_conversation:
        result = agent.process_input(msg)
        print(f"User: {msg}")
        print(f"Agent: {result['response'][:100]}{'...' if len(result['response']) > 100 else ''}")
        print(f"Classification: {result['classification']['intent']}")
        print(f"Routing: {result['routing']['department']} ({result['routing']['priority']})")
        print()
    
    print(agent.get_conversation_summary())
