"""
Feedback: Provides strategic points where human judgement is required using Azure OpenAI.
This component implements approval workflows and human-in-the-loop processes for high-risk decisions or complex judgments.

Azure OpenAI Documentation: https://docs.microsoft.com/en-us/azure/cognitive-services/openai/
"""

import os
import json
from typing import Optional, Dict, List, Callable
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from pydantic import BaseModel, Field

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


class ApprovalRequest(BaseModel):
    """Schema for approval requests."""
    id: str = Field(description="Unique identifier for the request")
    content: str = Field(description="Content to be reviewed")
    type: str = Field(description="Type of content (email, document, response, etc.)")
    risk_level: str = Field(description="Risk level: low, medium, high, critical")
    requester: str = Field(description="Who requested the approval")
    timestamp: str = Field(description="When the request was created")
    context: Optional[str] = Field(None, description="Additional context for the approval")


class ApprovalDecision(BaseModel):
    """Schema for approval decisions."""
    approved: bool = Field(description="Whether the content was approved")
    feedback: Optional[str] = Field(None, description="Human feedback or comments")
    modifications: Optional[str] = Field(None, description="Suggested modifications")
    approver: str = Field(description="Who made the approval decision")
    timestamp: str = Field(description="When the decision was made")


def get_human_approval(content: str, content_type: str = "general") -> bool:
    """
    Get human approval for content with enhanced interface.
    
    Args:
        content: Content to be reviewed
        content_type: Type of content for context
        
    Returns:
        Boolean approval decision
    """
    print("=" * 60)
    print(f"APPROVAL REQUEST - {content_type.upper()}")
    print("=" * 60)
    print(f"Content to review:\n{content}\n")
    print("=" * 60)
    
    while True:
        response = input("Approve this content? (y)es / (n)o / (m)odify / (v)iew again: ").lower().strip()
        
        if response.startswith('y'):
            return True
        elif response.startswith('n'):
            return False
        elif response.startswith('m'):
            modification = input("What modifications would you suggest? ")
            print(f"Modifications noted: {modification}")
            return False
        elif response.startswith('v'):
            print(f"\nContent:\n{content}\n")
            continue
        else:
            print("Please enter 'y' for yes, 'n' for no, 'm' for modify, or 'v' to view again.")


def assess_content_risk(content: str) -> str:
    """
    Assess the risk level of content using Azure OpenAI.
    
    Args:
        content: Content to assess
        
    Returns:
        Risk level (low, medium, high, critical)
    """
    client = get_azure_openai_client()
    deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
    
    risk_assessment_prompt = f"""
    Assess the risk level of this content for public or customer communication:
    
    Content: "{content}"
    
    Consider:
    - Potential for misunderstanding
    - Legal or compliance issues
    - Brand reputation impact
    - Sensitive information disclosure
    - Emotional or controversial topics
    
    Return only one word: low, medium, high, or critical
    """
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": risk_assessment_prompt}],
            temperature=0.1,
            max_tokens=50,
        )
        
        risk_level = response.choices[0].message.content.strip().lower()
        
        # Validate response
        if risk_level in ["low", "medium", "high", "critical"]:
            return risk_level
        else:
            return "medium"  # Default fallback
            
    except Exception as e:
        print(f"Risk assessment failed: {e}")
        return "high"  # Conservative default


def intelligence_with_human_feedback(prompt: str, require_approval: bool = True) -> dict:
    """
    Generate content with Azure OpenAI and optional human approval workflow.
    
    Args:
        prompt: Input prompt for content generation
        require_approval: Whether to require human approval
        
    Returns:
        Result dictionary with content and approval status
    """
    client = get_azure_openai_client()
    deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
    
    try:
        # Generate initial content
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
        )
        
        draft_content = response.choices[0].message.content
        
        result = {
            "content": draft_content,
            "generated_at": datetime.now().isoformat(),
            "prompt": prompt
        }
        
        if require_approval:
            # Assess risk level
            risk_level = assess_content_risk(draft_content)
            result["risk_level"] = risk_level
            
            # Determine if human approval is needed based on risk
            needs_approval = risk_level in ["high", "critical"]
            
            if needs_approval:
                approved = get_human_approval(draft_content, "AI Generated Content")
                result["approved"] = approved
                result["approval_required"] = True
                
                if approved:
                    result["status"] = "approved"
                    print("✅ Content approved for use")
                else:
                    result["status"] = "rejected"
                    print("❌ Content rejected")
            else:
                result["approved"] = True
                result["approval_required"] = False
                result["status"] = "auto_approved"
                print(f"✅ Content auto-approved (risk level: {risk_level})")
        else:
            result["approved"] = True
            result["approval_required"] = False
            result["status"] = "no_approval_needed"
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error",
            "approved": False
        }


class ApprovalWorkflow:
    """
    Advanced approval workflow manager with different approval strategies.
    """
    
    def __init__(self):
        self.client = get_azure_openai_client()
        self.deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT", "gpt-4o")
        self.pending_approvals: List[ApprovalRequest] = []
        self.approval_history: List[Dict] = []
    
    def submit_for_approval(
        self, 
        content: str, 
        content_type: str = "general",
        requester: str = "system",
        context: Optional[str] = None
    ) -> str:
        """
        Submit content for approval workflow.
        
        Args:
            content: Content to be approved
            content_type: Type of content
            requester: Who is requesting approval
            context: Additional context
            
        Returns:
            Approval request ID
        """
        risk_level = assess_content_risk(content)
        
        request_id = f"req_{len(self.pending_approvals) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        approval_request = ApprovalRequest(
            id=request_id,
            content=content,
            type=content_type,
            risk_level=risk_level,
            requester=requester,
            timestamp=datetime.now().isoformat(),
            context=context
        )
        
        self.pending_approvals.append(approval_request)
        
        print(f"📋 Approval request {request_id} submitted (Risk: {risk_level})")
        return request_id
    
    def process_approval(self, request_id: str, approver: str = "human") -> Optional[ApprovalDecision]:
        """
        Process a specific approval request.
        
        Args:
            request_id: ID of the approval request
            approver: Who is processing the approval
            
        Returns:
            Approval decision or None if request not found
        """
        # Find the request
        request = None
        for req in self.pending_approvals:
            if req.id == request_id:
                request = req
                break
        
        if not request:
            print(f"❌ Approval request {request_id} not found")
            return None
        
        # Display context
        print(f"\n📋 Processing approval request: {request_id}")
        print(f"Type: {request.type}")
        print(f"Risk Level: {request.risk_level}")
        print(f"Requester: {request.requester}")
        if request.context:
            print(f"Context: {request.context}")
        print(f"Submitted: {request.timestamp}")
        
        # Get approval decision
        approved = get_human_approval(request.content, request.type)
        
        feedback = None
        modifications = None
        
        if not approved:
            feedback = input("Please provide feedback (optional): ").strip()
            modifications = input("Suggest modifications (optional): ").strip()
        
        decision = ApprovalDecision(
            approved=approved,
            feedback=feedback if feedback else None,
            modifications=modifications if modifications else None,
            approver=approver,
            timestamp=datetime.now().isoformat()
        )
        
        # Record decision and remove from pending
        self.approval_history.append({
            "request": request.model_dump(),
            "decision": decision.model_dump()
        })
        
        self.pending_approvals.remove(request)
        
        return decision
    
    def process_all_pending(self, approver: str = "human") -> List[ApprovalDecision]:
        """
        Process all pending approval requests.
        
        Args:
            approver: Who is processing the approvals
            
        Returns:
            List of approval decisions
        """
        decisions = []
        
        while self.pending_approvals:
            request = self.pending_approvals[0]
            decision = self.process_approval(request.id, approver)
            if decision:
                decisions.append(decision)
        
        return decisions
    
    def get_pending_count(self) -> int:
        """Get number of pending approval requests."""
        return len(self.pending_approvals)
    
    def get_approval_stats(self) -> Dict:
        """Get approval statistics."""
        if not self.approval_history:
            return {"total": 0, "approved": 0, "rejected": 0, "approval_rate": 0.0}
        
        total = len(self.approval_history)
        approved = sum(1 for item in self.approval_history if item["decision"]["approved"])
        rejected = total - approved
        approval_rate = approved / total if total > 0 else 0.0
        
        return {
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approval_rate
        }


def batch_content_approval(content_items: List[str], content_type: str = "general") -> List[Dict]:
    """
    Process multiple content items through approval workflow.
    
    Args:
        content_items: List of content to approve
        content_type: Type of content
        
    Returns:
        List of approval results
    """
    workflow = ApprovalWorkflow()
    results = []
    
    # Submit all items for approval
    request_ids = []
    for i, content in enumerate(content_items):
        request_id = workflow.submit_for_approval(
            content, 
            content_type, 
            f"batch_job_{i+1}"
        )
        request_ids.append(request_id)
    
    print(f"\n📋 {len(content_items)} items submitted for approval")
    print("Processing approvals...\n")
    
    # Process all approvals
    decisions = workflow.process_all_pending()
    
    # Compile results
    for i, (content, decision) in enumerate(zip(content_items, decisions)):
        results.append({
            "content": content,
            "decision": decision.model_dump(),
            "index": i + 1
        })
    
    return results


if __name__ == "__main__":
    print("=== Feedback Demo ===\n")
    
    # Test 1: Simple approval workflow
    print("1. Testing simple approval workflow...")
    result1 = intelligence_with_human_feedback("Write a short poem about technology")
    print(f"Result: {result1}\n")
    
    # Test 2: High-risk content that should require approval
    print("2. Testing high-risk content approval...")
    result2 = intelligence_with_human_feedback(
        "Write a press release about our company's recent data breach and customer privacy concerns",
        require_approval=True
    )
    print(f"Result: {result2}\n")
    
    # Test 3: Approval workflow manager
    print("3. Testing approval workflow manager...")
    workflow = ApprovalWorkflow()
    
    # Submit some items for approval
    req1 = workflow.submit_for_approval(
        "Thank you for your purchase! Your order will arrive in 2-3 business days.",
        "customer_email",
        "customer_service"
    )
    
    req2 = workflow.submit_for_approval(
        "We're sorry to inform you that your account has been temporarily suspended due to suspicious activity.",
        "account_notification",
        "security_team",
        "Automated fraud detection triggered"
    )
    
    print(f"Pending approvals: {workflow.get_pending_count()}")
    
    # Process one approval
    print(f"\nProcessing approval {req1}...")
    decision1 = workflow.process_approval(req1)
    
    print(f"Decision: {decision1.model_dump_json(indent=2) if decision1 else 'None'}")
    
    # Show stats
    stats = workflow.get_approval_stats()
    print(f"Approval stats: {stats}")
    
    # Test 4: Batch approval (commented out for demo, uncomment to test)
    # print("4. Testing batch approval...")
    # batch_items = [
    #     "Welcome to our service!",
    #     "Your payment has been processed successfully.",
    #     "We need to discuss your account status urgently."
    # ]
    # batch_results = batch_content_approval(batch_items, "customer_communications")
    # print(f"Batch results: {len(batch_results)} items processed")
