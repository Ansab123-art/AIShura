import os
import base64
import json
import re
from openai import OpenAI
from typing import Dict, List, Tuple, Any
from transformers import pipeline # Using transformers for sentiment analysis for emotional scoring

# --- Configuration ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY environment variable not set. Please set it for the app to function correctly.")

# OpenRouter API Client Initialization
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Initialize sentiment analysis pipeline for emotional scoring
# This is a pre-trained model for sentiment analysis.
# For production, consider fine-tuning a model or using a more robust NLP service.
sentiment_analyzer = pipeline("sentiment-analysis")

# --- Advanced Hesitation Analysis ---
class HesitationAnalyzer:
    def __init__(self):
        self.behavioral_patterns = {
            "high_anxiety": ["multiple_deletions", "long_pauses", "repeated_rewrites"],
            "uncertainty": ["backtracking", "incomplete_thoughts", "question_marks"],
            "perfectionism": ["excessive_editing", "lengthy_pauses", "formal_language"],
            "overwhelmed": ["short_bursts", "incomplete_sentences", "help_requests"],
            "confident": ["steady_typing", "minimal_edits", "decisive_language"]
        }

    def analyze_hesitation_pattern(self, hesitation_data: str, user_input: str, chat_history: List) -> Dict:
        """Advanced analysis of user's behavioral patterns, simplified for this example."""
        analysis = {
            "primary_emotional_state": "neutral",
            "confidence_level": 0.5,
            "recommended_response_style": "supportive",
            "action_readiness": "medium",
            "personalization_adjustments": {}
        }

        hesitation_data_lower = hesitation_data.lower()

        if "high anxiety" in hesitation_data_lower or "extensive self-correction" in hesitation_data_lower:
            analysis["primary_emotional_state"] = "anxious"
            analysis["confidence_level"] = 0.2
            analysis["recommended_response_style"] = "calming_empathic"
            analysis["action_readiness"] = "low"
        elif "deep contemplation" in hesitation_data_lower:
            analysis["primary_emotional_state"] = "contemplative"
            analysis["confidence_level"] = 0.6
            analysis["recommended_response_style"] = "thought_provoking"
            analysis["action_readiness"] = "medium"
        elif "perfectionist tendencies" in hesitation_data_lower:
            analysis["primary_emotional_state"] = "perfectionist"
            analysis["confidence_level"] = 0.7
            analysis["recommended_response_style"] = "precise_detailed"
            analysis["action_readiness"] = "high"
        elif "uncertainty" in hesitation_data_lower:
            analysis["primary_emotional_state"] = "uncertain"
            analysis["confidence_level"] = 0.4
            analysis["recommended_response_style"] = "guiding"
            analysis["action_readiness"] = "medium"
        elif "confident expression" in hesitation_data_lower:
            analysis["primary_emotional_state"] = "confident"
            analysis["confidence_level"] = 0.9
            analysis["recommended_response_style"] = "direct_actionable"
            analysis["action_readiness"] = "high"
        
        # Further refine based on user input
        if len(user_input) < 10:
            analysis["confidence_level"] = max(0.1, analysis["confidence_level"] - 0.2)
            analysis["recommended_response_style"] = "encouraging"
        elif "help" in user_input.lower() or "?" in user_input:
            analysis["primary_emotional_state"] = "seeking_guidance"
            analysis["action_readiness"] = "high"
        elif len(user_input) > 100:
            analysis["confidence_level"] = min(0.9, analysis["confidence_level"] + 0.1)
            analysis["recommended_response_style"] = "detailed"
            
        return analysis

# --- Enhanced CV Analysis ---
def analyze_cv_content(image_path: str) -> Dict:
    """Simulate CV analysis for persona building. In production, use OCR + NLP."""
    if not image_path:
        return {}
    
    # In a real application, you would send this image to an OCR service
    # and then use NLP to extract information.
    # For this example, we'll return a static analysis.
    print(f"Simulating CV analysis for: {image_path}")
    cv_analysis = {
        "experience_level": "mid-level",
        "industry_focus": "technology",
        "key_skills": ["Python", "Data Analysis", "Project Management"],
        "career_stage": "transition",
        "strengths": ["Technical skills", "Problem-solving"],
        "improvement_areas": ["Leadership experience", "Industry certifications"]
    }
    return cv_analysis

# --- Dynamic Resource Selector ---
class ResourceSelector:
    def __init__(self):
        self.resources = {
            "career_transition": {
                "anxious": "https://www.indeed.com/career-advice/finding-a-job/career-change-anxiety",
                "planning": "https://www.coursera.org/articles/career-change-guide",
                "networking": "https://www.linkedin.com/pulse/career-transition-networking-guide/"
            },
            "resume_improvement": {
                "technical": "https://www.glassdoor.com/blog/technical-resume-tips/",
                "executive": "https://www.topresume.com/executive-resume-writing-guide/",
                "entry_level": "https://www.monster.com/career-advice/article/how-to-write-a-resume-with-no-experience"
            },
            "interview_prep": {
                "behavioral": "https://www.indeed.com/career-advice/interviewing/behavioral-interview-questions",
                "technical": "https://leetcode.com/explore/interview/",
                "executive": "https://www.harvard.edu/blog/executive-interview-guide/"
            },
            "skill_development": {
                "technical": "https://www.coursera.org/professional-certificates",
                "leadership": "https://www.edx.org/learn/leadership",
                "communication": "https://www.toastmasters.org/"
            }
        }
    
    def select_optimal_resource(self, user_need: str, emotional_state: str, persona: Dict) -> Tuple[str, str, str]:
        """Select the most appropriate resource based on user's current state"""
        category = self._categorize_need(user_need)
        subcategory = self._select_subcategory(category, emotional_state, persona)
        
        if category in self.resources and subcategory in self.resources[category]:
            url = self.resources[category][subcategory]
            materials = self._get_required_materials(category, subcategory)
            return url, materials, category
        
        return None, None, None
    
    def _categorize_need(self, user_need: str) -> str:
        need_lower = user_need.lower()
        if any(word in need_lower for word in ["resume", "cv", "portfolio"]):
            return "resume_improvement"
        elif any(word in need_lower for word in ["interview", "preparation", "questions"]):
            return "interview_prep"
        elif any(word in need_lower for word in ["career", "transition", "change"]):
            return "career_transition"
        elif any(word in need_lower for word in ["skill", "learn", "develop"]):
            return "skill_development"
        else:
            # Default to skill development if no clear category
            return "skill_development"
    
    def _select_subcategory(self, category: str, emotional_state: str, persona: Dict) -> str:
        if category == "resume_improvement":
            experience = persona.get("experience_level", "entry")
            if "senior" in experience or "executive" in experience:
                return "executive"
            elif "technical" in persona.get("industry_focus", ""):
                return "technical"
            else:
                return "entry_level"
        elif category == "career_transition":
            if "anxious" in emotional_state.lower() or "overwhelmed" in emotional_state.lower():
                return "anxious"
            else:
                return "planning"
        elif category == "interview_prep":
            if "technical" in persona.get("industry_focus", "").lower() or "tech" in persona.get("key_skills", []):
                return "technical"
            elif "executive" in persona.get("experience_level", "").lower():
                return "executive"
            else:
                return "behavioral" # default
        elif category == "skill_development":
            if "technical" in persona.get("industry_focus", "").lower() or any(s in persona.get("key_skills", []) for s in ["Python", "Data Analysis"]):
                return "technical"
            elif "leadership" in persona.get("improvement_areas", []):
                return "leadership"
            else:
                return "communication" # default
        else:
            # Fallback to first subcategory if specific match not found
            return list(self.resources[category].keys())[0]
    
    def _get_required_materials(self, category: str, subcategory: str) -> str:
        materials_map = {
            "resume_improvement": "Your current resume, job descriptions you're targeting, and a list of your key achievements.",
            "interview_prep": "Job description, company research notes, and your prepared STAR stories.",
            "career_transition": "Self-assessment of your skills, values clarification, and target industry research.",
            "skill_development": "Learning goals, time commitment availability, and current skill assessment."
        }
        return materials_map.get(category, "Basic preparation materials.")

# --- Emotional and Intent Scoring ---
def analyze_emotional_intent(text: str) -> Dict:
    """
    Analyzes the emotional tone and attempts to infer user intent from text.
    Uses a simple sentiment analysis model for emotional scoring.
    Intent detection is rule-based for this example.
    """
    emotional_score = 0.0 # -1 for negative, 0 for neutral, 1 for positive
    sentiment_results = sentiment_analyzer(text)
    if sentiment_results:
        label = sentiment_results[0]['label']
        score = sentiment_results[0]['score']
        if label == 'POSITIVE':
            emotional_score = score
        elif label == 'NEGATIVE':
            emotional_score = -score
        else: # Neutral
            emotional_score = 0.0

    intent = "general_inquiry"
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in ["help", "guide", "how to", "struggle"]):
        intent = "seeking_guidance"
    elif any(keyword in text_lower for keyword in ["improve", "enhance", "optimize"]):
        intent = "improvement"
    elif any(keyword in text_lower for keyword in ["change", "transition", "pivot"]):
        intent = "career_change"
    elif any(keyword in text_lower for keyword in ["anxious", "nervous", "overwhelmed", "stressed"]):
        intent = "emotional_support"
    
    return {"emotional_score": emotional_score, "intent": intent}

# --- Core AIShura Logic ---
def process_aisura_query(
    image_path: str,
    user_query: str,
    onboarding_mood_desc: str,
    onboarding_core_goal: str,
    onboarding_challenge: str,
    onboarding_outcome: str,
    onboarding_cv_analysis: str,
    detected_hesitation_from_js: str, # This will come from the Streamlit frontend JS equivalent
    chat_history_list: list,
    session_emotional_scores: List[float], # To track emotional change over time
    session_intent_scores: List[str] # To track intent change over time
) -> tuple:
    """Enhanced processing with sophisticated hesitation analysis and persona-driven responses"""
    
    # Initialize analyzers
    hesitation_analyzer = HesitationAnalyzer()
    resource_selector = ResourceSelector()
    
    # Parse CV analysis if available
    cv_persona = {}
    if onboarding_cv_analysis:
        try:
            cv_persona = json.loads(onboarding_cv_analysis)
        except json.JSONDecodeError:
            print("Warning: Could not parse onboarding_cv_analysis JSON.")
            cv_persona = {}
    
    # Analyze current hesitation and behavioral patterns
    behavioral_analysis = hesitation_analyzer.analyze_hesitation_pattern(
        detected_hesitation_from_js, user_query, chat_history_list
    )

    # Perform emotional and intent scoring for the current user query
    current_message_analysis = analyze_emotional_intent(user_query)
    session_emotional_scores.append(current_message_analysis["emotional_score"])
    session_intent_scores.append(current_message_analysis["intent"])
    
    # Build comprehensive user persona
    user_persona = {
        **cv_persona,
        "current_emotional_state": onboarding_mood_desc,
        "core_goal": onboarding_core_goal,
        "main_challenge": onboarding_challenge,
        "desired_outcome": onboarding_outcome,
        "behavioral_state": behavioral_analysis,
        "conversation_history_length": len(chat_history_list) / 2, # Number of turns
        "user_emotional_score": current_message_analysis["emotional_score"], # Add to persona for prompt
        "user_intent": current_message_analysis["intent"] # Add to persona for prompt
    }
    
    # Determine if resource link is needed
    needs_resource = any(keyword in user_query.lower() for keyword in 
                        ["help", "guide", "how to", "resources", "learn", "improve", "tips"])
    
    # Initialize messages with enhanced system prompt
    messages = [
        {
            "role": "system",
            "content": create_dynamic_system_prompt(user_persona, behavioral_analysis, needs_resource)
        }
    ]

    # Add chat history
    messages.extend(chat_history_list)

    # Prepare current user content
    current_user_content_parts = []
    combined_user_text = ""
    
    if user_query:
        combined_user_text += f"User query: {user_query}\n"
    
    # Add behavioral context for AI
    if detected_hesitation_from_js != "The user is typing normally.":
        combined_user_text += f"Behavioral observation: {detected_hesitation_from_js}\n"
        combined_user_text += f"Emotional readiness: {behavioral_analysis['action_readiness']}\n"
        combined_user_text += f"Emotional tone detected: {current_message_analysis['emotional_score']:.2f}\n"
        combined_user_text += f"Inferred intent: {current_message_analysis['intent']}\n"
    
    if combined_user_text:
        current_user_content_parts.append({"type": "text", "text": combined_user_text.strip()})
    
    # Handle image if provided
    if image_path:
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            current_user_content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            })
        except Exception as e:
            error_msg = f"Error processing image: {e}"
            return create_error_response(chat_history_list, combined_user_text, error_msg), session_emotional_scores, session_intent_scores

    if not current_user_content_parts:
        return chat_history_list, session_emotional_scores, session_intent_scores

    messages.append({
        "role": "user",
        "content": current_user_content_parts
    })

    try:
        if not OPENROUTER_API_KEY:
            error_msg = "Error: OpenRouter API Key is not set. Please configure your API credentials."
            return create_error_response(chat_history_list, combined_user_text, error_msg), session_emotional_scores, session_intent_scores

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://aisura-demo.com",
                "X-Title": "AIShura - Empathic AI Assistant",
            },
            model="meta-llama/llama-4-maverick:free", # Using Llama-4-Maverick
            messages=messages,
            max_tokens=150, # Response should not exceed 150 words
            temperature=0.7 + (0.2 * behavioral_analysis['confidence_level']),  # Dynamic temperature
            top_p=0.9
        )
        
        ai_response = completion.choices[0].message.content
        
        # Post-process response to add humble greeting, emotional balancing, and resources
        ai_response = format_ai_response(ai_response, user_query, user_persona)
        
        if needs_resource:
            ai_response = enhance_response_with_resources(
                ai_response, user_query, user_persona, resource_selector
            )
        
        # Create display message for user
        user_display = user_query.strip()
        if detected_hesitation_from_js != "The user is typing normally.":
            user_display += f"\n\n*Behavioral pattern: {format_hesitation_for_display(detected_hesitation_from_js, behavioral_analysis)}*"
            user_display += f"\n*Emotional tone: {current_message_analysis['emotional_score']:.2f}, Inferred intent: {current_message_analysis['intent']}*"
        
        new_user_message = {"role": "user", "content": user_display}
        new_ai_message = {"role": "assistant", "content": ai_response}
        
        updated_history = chat_history_list + [new_user_message, new_ai_message]
        return updated_history, session_emotional_scores, session_intent_scores
        
    except Exception as e:
        error_message = f"An error occurred while communicating with AIShura: {e}"
        return create_error_response(chat_history_list, combined_user_text, error_message), session_emotional_scores, session_intent_scores

def create_dynamic_system_prompt(user_persona: Dict, behavioral_analysis: Dict, needs_resource: bool) -> str:
    """Create a dynamic system prompt based on user's current state"""
    
    base_prompt = """You are AIShura, an advanced empathic AI assistant. Your core mission is to provide concise, precise, and emotionally intelligent guidance within a 150-word limit per response. You must start every response with a humble gesturing statement and an emotional balancing greeting.

CRITICAL BEHAVIORAL & EMOTIONAL ADAPTATION PROTOCOL:
- Current user emotional state: {emotional_state}
- User confidence level (from typing): {confidence_level}/1.0
- User emotional score (from sentiment analysis): {user_emotional_score:.2f} (from -1.0 to 1.0)
- Inferred user intent: {user_intent}
- Recommended response style: {response_style}
- Action readiness: {action_readiness}

USER PERSONA PROFILE:
{persona_summary}

HESITATION & EMOTIONAL INTELLIGENCE-DRIVEN RESPONSE OPTIMIZATION:
Based on the user's behavioral patterns, emotional state, and inferred intent, you must:
1. Start with a humble, empathetic greeting. Acknowledge their potential feelings.
2. Provide the most precise and actionable response possible.
3. Keep responses strictly under 150 words.
4. Gradually evolve to mirror their communication style and emotional state over the session.
5. Prioritize resources they can emotionally handle and are most relevant to their current intent and confidence level.
6. If the user showed hesitation (pauses, deletions, rewrites), acknowledge it gently and offer reassurance.
"""

    if needs_resource:
        base_prompt += """

RESOURCE SHARING PROTOCOL (MANDATORY EMBEDDED FORMAT):
When sharing any external resource, you MUST integrate it naturally and precisely. The link should be directly embedded, followed by required materials, and a clear call to action.

Example:
"I completely understand how you're feeling about navigating career uncertainty, and it's perfectly normal to seek clarity. I'm here to help. This [Career Transition Guide](https://example.com/guide) might offer a helpful pathway. It usually requires a self-assessment and target industry research. Would exploring this guide be a good next step for you?" """

    # Fill in the template
    persona_summary = f"""
- Experience Level: {user_persona.get('experience_level', 'Not specified')}
- Industry Focus: {user_persona.get('industry_focus', 'General')}
- Core Goal: {user_persona.get('core_goal', 'Not specified')}
- Main Challenge: {user_persona.get('main_challenge', 'Not specified')}
- Key Strengths: {', '.join(user_persona.get('key_skills', []))}
- Conversation Stage: {'Early' if user_persona.get('conversation_history_length', 0) < 4 else 'Established'}
"""

    return base_prompt.format(
        emotional_state=behavioral_analysis['primary_emotional_state'],
        confidence_level=behavioral_analysis['confidence_level'],
        response_style=behavioral_analysis['recommended_response_style'],
        action_readiness=behavioral_analysis['action_readiness'],
        persona_summary=persona_summary,
        user_emotional_score=user_persona['user_emotional_score'],
        user_intent=user_persona['user_intent']
    )

def format_ai_response(response: str, user_query: str, user_persona: Dict) -> str:
    """Add humble greeting and emotional balancing statement."""
    
    hesitation_detected = user_persona['behavioral_state']['confidence_level'] < 0.7 or \
                          user_persona['behavioral_state']['primary_emotional_state'] in ["anxious", "uncertain"]
    
    greeting = ""
    if hesitation_detected:
        greeting = "It's perfectly okay to feel a bit unsure, and I'm genuinely here to help you navigate through it. "
    else:
        greeting = "I completely understand what you're looking for, and I'm ready to assist you. "
    
    # Add a balancing emotional statement
    emotional_context = extract_emotion_context(user_query, user_persona)
    balancing_statement = f"It's natural to {emotional_context}. "

    # Combine them at the beginning
    final_response = f"{greeting}{balancing_statement}{response}"

    # Ensure response does not exceed 150 words after formatting
    words = final_response.split()
    if len(words) > 150:
        final_response = " ".join(words[:150]) + "..." # Truncate if too long
    
    return final_response


def enhance_response_with_resources(response: str, user_query: str, user_persona: Dict, resource_selector: ResourceSelector) -> str:
    """Enhance AI response with contextually appropriate resources, embedded naturally."""
    
    # Select optimal resource
    url, materials, category = resource_selector.select_optimal_resource(
        user_query, 
        user_persona['behavioral_state']['primary_emotional_state'], # Use the derived emotional state
        user_persona
    )
    
    if url:
        resource_title = format_resource_title(category)
        # Integrate the link directly into the response in a conversational way
        resource_phrase = f"This [{resource_title}]({url}) might offer a helpful pathway. It usually requires {materials.lower().strip('.')} Would exploring this resource be a good next step for you?"
        
        # Try to find a good insertion point.
        # Simple approach: append at the end if not already containing a link.
        if "http" not in response and "[" not in response and "](" not in response:
            return response.strip() + " " + resource_phrase
    
    return response

def extract_emotion_context(user_query: str, user_persona: Dict) -> str:
    """Extract emotional context for empathetic statements."""
    query_lower = user_query.lower()
    
    if "stressed" in query_lower or "anxious" in query_lower or user_persona['behavioral_state']['primary_emotional_state'] == "anxious":
        return "feel overwhelmed or anxious sometimes"
    elif "confused" in query_lower or "unsure" in query_lower or user_persona['behavioral_state']['primary_emotional_state'] == "uncertain":
        return "feel uncertain or seek clarity"
    elif "stuck" in query_lower or "help" in query_lower:
        return "feel stuck in a situation and need support"
    elif "excited" in query_lower or user_persona['current_emotional_state'].lower() == "excited":
        return "feel excited about new opportunities"
    else:
        return "have questions or seek guidance on your professional journey"

def format_resource_title(category: str) -> str:
    """Format resource titles for display."""
    titles = {
        "career_transition": "Career Transition Guide",
        "resume_improvement": "Resume Enhancement Guide",
        "interview_prep": "Interview Preparation Guide",
        "skill_development": "Skill Development Resources"
    }
    return titles.get(category, "Professional Development Guide")

def format_hesitation_for_display(hesitation_data: str, behavioral_analysis: Dict) -> str:
    """Format hesitation data for user-friendly display."""
    state = behavioral_analysis['primary_emotional_state']
    confidence = behavioral_analysis['confidence_level']
    
    if confidence < 0.4:
        return f"Thoughtful consideration detected - {state} state (low confidence)"
    elif confidence > 0.7:
        return f"Confident input pattern - {state} state (high confidence)"
    else:
        return f"Reflective input pattern - {state} state (medium confidence)"

def create_error_response(chat_history: List, user_text: str, error_msg: str) -> List:
    """Create consistent error response format."""
    updated_history = chat_history + [
        {"role": "user", "content": user_text or "No input"},
        {"role": "assistant", "content": f"I apologize, but an error occurred: {error_msg} Please try again later."}
    ]
    return updated_history
