"""
Orchestrator-Workers with Azure OpenAI.
Demonstrates using a central LLM to dynamically analyze tasks, coordinate specialized workers,
and synthesize their results for complex content creation workflows.
"""

import sys
import os
import logging
from typing import List, Dict

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
# Step 1: Define the data models
# --------------------------------------------------------------

class SubTask(BaseModel):
    """Blog section task defined by orchestrator"""
    
    section_type: str = Field(description="Type of blog section to write")
    description: str = Field(description="What this section should cover")
    style_guide: str = Field(description="Writing style for this section")
    target_length: int = Field(description="Target word count for this section")


class OrchestratorPlan(BaseModel):
    """Orchestrator's blog structure and tasks"""
    
    topic_analysis: str = Field(description="Analysis of the blog topic")
    target_audience: str = Field(description="Intended audience for the blog")
    sections: List[SubTask] = Field(description="List of sections to write")


class SectionContent(BaseModel):
    """Content written by a worker"""
    
    content: str = Field(description="Written content for the section")
    key_points: List[str] = Field(description="Main points covered")
    word_count: int = Field(description="Actual word count of the content")


class SuggestedEdits(BaseModel):
    """Suggested edits for a section"""
    
    section_name: str = Field(description="Name of the section")
    suggested_edit: str = Field(description="Suggested edit")
    priority: str = Field(description="Priority level: low, medium, high")


class ReviewFeedback(BaseModel):
    """Final review and suggestions"""
    
    cohesion_score: float = Field(description="How well sections flow together (0-1)")
    suggested_edits: List[SuggestedEdits] = Field(
        description="Suggested edits by section"
    )
    final_version: str = Field(description="Complete, polished blog post")
    overall_quality: str = Field(description="Overall quality assessment")


# --------------------------------------------------------------
# Step 2: Define Azure OpenAI-optimized prompts
# --------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """
You are an expert content strategist and blog orchestrator. Your role is to analyze blog topics and create comprehensive content plans.

Analyze the given topic and break it down into logical, flowing sections that will create an engaging and informative blog post.

Consider:
- Narrative flow and logical progression
- Target audience needs and interests
- Content depth and complexity
- Section interdependencies
"""

WORKER_SYSTEM_PROMPT = """
You are a skilled technical writer specializing in creating engaging, well-structured content.

Your task is to write high-quality blog sections that:
- Follow the specified style guide precisely
- Maintain consistency with previous sections
- Meet the target word count
- Include clear, actionable insights
- Use engaging, readable language
"""

REVIEWER_SYSTEM_PROMPT = """
You are an expert content editor and quality assurance specialist.

Your role is to:
- Evaluate overall cohesion and flow between sections
- Identify areas for improvement
- Ensure consistent tone and style
- Create a polished, publication-ready final version
- Provide constructive, specific feedback
"""


# --------------------------------------------------------------
# Step 3: Implement Azure OpenAI Blog Orchestrator
# --------------------------------------------------------------

class AzureBlogOrchestrator:
    """Blog writing orchestrator using Azure OpenAI."""
    
    def __init__(self):
        self.client = get_azure_openai_client()
        self.deployment_name = get_deployment_name("gpt4")
        self.sections_content = {}
        logger.info("Azure Blog Orchestrator initialized")

    def get_plan(self, topic: str, target_length: int, style: str) -> OrchestratorPlan:
        """Get orchestrator's blog structure plan using Azure OpenAI."""
        logger.info("Orchestrator analyzing topic and creating plan")
        
        user_prompt = f"""
        Create a comprehensive blog plan for:
        
        Topic: {topic}
        Target Length: {target_length} words
        Style: {style}
        
        Provide a detailed analysis of the topic, define the target audience, and break down the content into logical sections with specific guidance for each section.
        """
        
        completion = self.client.beta.chat.completions.parse(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format=OrchestratorPlan,
            temperature=0.3  # Balanced creativity for planning
        )
        
        plan = completion.choices[0].message.parsed
        logger.info(f"Plan created with {len(plan.sections)} sections")
        return plan

    def write_section(self, topic: str, section: SubTask, previous_context: str = "") -> SectionContent:
        """
        Worker: Write a specific blog section with context from previous sections.
        
        Args:
            topic: The main blog topic
            section: SubTask containing section details
            previous_context: Context from previously written sections
            
        Returns:
            SectionContent: The written content and key points
        """
        logger.info(f"Worker writing section: {section.section_type}")
        
        context_text = f"\n\nPREVIOUS SECTIONS CONTEXT:\n{previous_context}" if previous_context else "\n\nThis is the first section of the blog post."
        
        user_prompt = f"""
        Write a blog section with the following specifications:
        
        Main Topic: {topic}
        Section Type: {section.section_type}
        Section Goal: {section.description}
        Style Guide: {section.style_guide}
        Target Word Count: {section.target_length} words
        {context_text}
        
        Create engaging, well-structured content that flows naturally with the overall blog narrative.
        """
        
        completion = self.client.beta.chat.completions.parse(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": WORKER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format=SectionContent,
            temperature=0.5  # Higher creativity for content writing
        )
        
        content = completion.choices[0].message.parsed
        logger.info(f"Section completed: {content.word_count} words, {len(content.key_points)} key points")
        return content

    def review_post(self, topic: str, plan: OrchestratorPlan) -> ReviewFeedback:
        """Reviewer: Analyze and improve overall cohesion using Azure OpenAI."""
        logger.info("Reviewer analyzing complete blog post")
        
        sections_text = "\n\n".join([
            f"=== {section_type.upper()} ===\n{content.content}\n\nKey Points: {', '.join(content.key_points)}"
            for section_type, content in self.sections_content.items()
        ])
        
        user_prompt = f"""
        Review this complete blog post for cohesion, flow, and overall quality:
        
        Topic: {topic}
        Target Audience: {plan.target_audience}
        
        COMPLETE BLOG CONTENT:
        {sections_text}
        
        Provide:
        1. A cohesion score (0.0-1.0) reflecting how well sections flow together
        2. Specific, actionable suggested edits for each section if needed
        3. An overall quality assessment
        4. A final, polished version incorporating improvements
        """
        
        completion = self.client.beta.chat.completions.parse(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format=ReviewFeedback,
            temperature=0.2  # Lower temperature for consistent review quality
        )
        
        review = completion.choices[0].message.parsed
        logger.info(f"Review completed - Cohesion score: {review.cohesion_score:.2f}")
        return review

    def write_blog(self, topic: str, target_length: int = 1000, style: str = "informative") -> Dict:
        """
        Process the entire blog writing task using the orchestrator pattern.
        
        Args:
            topic: Blog topic to write about
            target_length: Target word count for the complete blog
            style: Writing style (e.g., "technical but accessible", "conversational", "formal")
            
        Returns:
            Dict: Complete blog writing results including plan, sections, and review
        """
        logger.info(f"Starting Azure OpenAI blog writing process for: '{topic}'")
        
        try:
            # Phase 1: Planning (Orchestrator)
            plan = self.get_plan(topic, target_length, style)
            logger.info(f"Planning complete - Target audience: {plan.target_audience}")
            
            # Phase 2: Content Creation (Workers)
            previous_context = ""
            for i, section in enumerate(plan.sections):
                logger.info(f"Writing section {i+1}/{len(plan.sections)}: {section.section_type}")
                
                content = self.write_section(topic, section, previous_context)
                self.sections_content[section.section_type] = content
                
                # Build context for next section
                previous_context += f"\n\n{section.section_type}: {content.content[:200]}..."
            
            # Phase 3: Review and Polish (Reviewer)
            logger.info("Starting final review and polish phase")
            review = self.review_post(topic, plan)
            
            # Calculate total statistics
            total_words = sum(content.word_count for content in self.sections_content.values())
            total_sections = len(self.sections_content)
            
            result = {
                "topic": topic,
                "structure": plan,
                "sections": self.sections_content,
                "review": review,
                "statistics": {
                    "total_words": total_words,
                    "total_sections": total_sections,
                    "target_words": target_length,
                    "cohesion_score": review.cohesion_score
                }
            }
            
            logger.info(f"Blog writing completed successfully - {total_words} words, cohesion: {review.cohesion_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in blog writing process: {e}")
            raise


def demonstrate_orchestrator():
    """Demonstrate the orchestrator-workers pattern with Azure OpenAI."""
    
    print("Azure OpenAI Orchestrator-Workers Demo")
    print("=" * 60)
    
    try:
        orchestrator = AzureBlogOrchestrator()
        
        # Example topics with different styles
        test_cases = [
            {
                "topic": "The impact of AI on software development",
                "target_length": 1200,
                "style": "technical but accessible"
            },
            {
                "topic": "Best practices for remote team collaboration",
                "target_length": 800,
                "style": "conversational and practical"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"Test Case {i}: {test_case['topic']}")
            print(f"Target Length: {test_case['target_length']} words")
            print(f"Style: {test_case['style']}")
            print("="*60)
            
            result = orchestrator.write_blog(**test_case)
            
            # Display results
            print(f"\n📊 STATISTICS:")
            stats = result['statistics']
            print(f"   • Target Words: {stats['target_words']}")
            print(f"   • Actual Words: {stats['total_words']}")
            print(f"   • Sections: {stats['total_sections']}")
            print(f"   • Cohesion Score: {stats['cohesion_score']:.2f}/1.0")
            
            print(f"\n🎯 TARGET AUDIENCE:")
            print(f"   {result['structure'].target_audience}")
            
            print(f"\n📝 SECTIONS WRITTEN:")
            for section_type, content in result['sections'].items():
                print(f"   • {section_type}: {content.word_count} words")
            
            print(f"\n⭐ QUALITY ASSESSMENT:")
            print(f"   {result['review'].overall_quality}")
            
            if result['review'].suggested_edits:
                print(f"\n🔧 SUGGESTED IMPROVEMENTS:")
                for edit in result['review'].suggested_edits:
                    print(f"   • {edit.section_name} ({edit.priority}): {edit.suggested_edit}")
            
            print(f"\n📖 FINAL BLOG POST:")
            print("-" * 60)
            print(result['review'].final_version)
            
            # Reset for next test case
            orchestrator.sections_content = {}
            
            if i < len(test_cases):
                input("\nPress Enter to continue to next test case...")
    
    except Exception as e:
        print(f"❌ Error in demonstration: {e}")


def main():
    """Main demonstration of orchestrator-workers pattern with Azure OpenAI."""
    
    # Validate environment configuration
    if not validate_environment():
        print("Environment validation failed. Please check your .env configuration.")
        return
    
    demonstrate_orchestrator()


if __name__ == "__main__":
    main()
