import os
import base64
import json
import re
from openai import OpenAI
from typing import Dict, List, Tuple, Any

# --- Configuration ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY environment variable not set. Please set it for the app to function correctly.")

# OpenRouter API Client Initialization
# Ensure the client is accessible where needed
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- Advanced Hesitation Analysis ---
class HesitationAnalyzer:
    def __init__(self):
        # Behavioral patterns are conceptual for the AI's understanding
        self.behavioral_patterns = {
            "high_anxiety": ["multiple_deletions", "long_pauses", "repeated_rewrites"],
            "uncertainty": ["backtracking", "incomplete_thoughts", "question_marks"],
            "perfectionism": ["excessive_editing", "lengthy_pauses", "formal_language"],
            "overwhelmed": ["short_bursts", "incomplete_sentences", "help_requests"],
            "confident": ["steady_typing", "minimal_edits", "decisive_language"]
        }

    def analyze_hesitation_pattern(self, hesitation_data: str, user_input: str, chat_history: List) -> Dict:
        """
        Analyzes user's behavioral patterns based on simulated hesitation data.
        In a full Streamlit app, real-time JS would feed more accurate data.
        For demonstration, this uses a simplified parsing of the `hesitation_data` string.
        """
        analysis = {
            "primary_emotional_state": "neutral",
            "confidence_level": 0.5, # Default
            "recommended_response_style": "supportive",
            "action_readiness": "medium",
            "personalization_adjustments": {}
        }
        
        hesitation_data_lower = hesitation_data.lower()

        # Update confidence based on reported patterns
        if "low confidence" in hesitation_data_lower:
            analysis["confidence_level"] = 0.2
            analysis["primary_emotional_state"] = "anxious"
            analysis["recommended_response_style"] = "calming_empathic"
            analysis["action_readiness"] = "low"
        elif "high confidence" in hesitation_data_lower:
            analysis["confidence_level"] = 0.9
            analysis["primary_emotional_state"] = "confident"
            analysis["recommended_response_style"] = "direct_actionable"
            analysis["action_readiness"] = "high"
        elif "medium confidence" in hesitation_data_lower:
            analysis["confidence_level"] = 0.6
            analysis["primary_emotional_state"] = "reflective"
            analysis["recommended_response_style"] = "thought_provoking"
            analysis["action_readiness"] = "medium"

        # Further refine based on emotional state detected by the AI
        if "anxious" in hesitation_data_lower:
            analysis["primary_emotional_state"] = "anxious"
            analysis["recommended_response_style"] = "calming_empathic"
            analysis["action_readiness"] = "low"
        elif "uncertain" in hesitation_data_lower:
            analysis["primary_emotional_state"] = "uncertain"
            analysis["recommended_response_style"] = "guiding"
            analysis["action_readiness"] = "medium"
        elif "perfectionist" in hesitation_data_lower:
            analysis["primary_emotional_state"] = "perfectionist"
            analysis["recommended_response_style"] = "precise_detailed"
            analysis["action_readiness"] = "high"

        # Additional adjustments based on user input length/keywords
        if len(user_input) < 10:
            analysis["confidence_level"] = max(0.1, analysis["confidence_level"] - 0.1)
            analysis["recommended_response_style"] = "encouraging"
        elif "help" in user_input.lower() or "?" in user_input:
            analysis["action_readiness"] = "high" # User is actively seeking next steps
            
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
            },
            "workplace_issues": { # Added new category for workplace issues
                "discrimination": "https://www.eeoc.gov/discrimination-type",
                "stress_management": "https://www.apa.org/topics/stress/work",
                "conflict_resolution": "https://hbr.org/2021/01/how-to-resolve-conflict-at-work"
            },
            "job_search": { # Added for job search specific resources
                "remote_jobs": "https://weworkremotely.com/",
                "ai_ml_jobs": "https://www.builtin.com/jobs/ai-ml",
                "data_science_jobs": "https://www.datasciencecentral.com/jobs/"
            }
        }
    
    def select_optimal_resource(self, user_need: str, emotional_state: str, persona: Dict) -> Tuple[str, str, str]:
        """Select the most appropriate resource based on user's current state"""
        category = self._categorize_need(user_need, persona.get('user_intent', 'general_inquiry'))
        subcategory = self._select_subcategory(category, emotional_state, persona, user_need)
        
        if category in self.resources and subcategory in self.resources[category]:
            url = self.resources[category][subcategory]
            materials = self._get_required_materials(category, subcategory)
            return url, materials, category
        
        return None, None, None
    
    def _categorize_need(self, user_need: str, user_intent: str) -> str:
        need_lower = user_need.lower()
        if user_intent == "resume_improvement" or any(word in need_lower for word in ["resume", "cv", "portfolio"]):
            return "resume_improvement"
        elif user_intent == "interview_prep" or any(word in need_lower for word in ["interview", "preparation", "questions"]):
            return "interview_prep"
        elif user_intent == "career_change" or any(word in need_lower for word in ["career", "transition", "change", "pivot"]):
            return "career_transition"
        elif user_intent == "skill_development" or any(word in need_lower for word in ["skill", "learn", "develop"]):
            return "skill_development"
        elif user_intent == "workplace_issues" or any(word in need_lower for word in ["discrimination", "harassment", "stress", "conflict", "toxic"]):
            return "workplace_issues"
        elif user_intent == "job_search" or any(word in need_lower for word in ["job search", "remote jobs", "ai/ml jobs", "data science jobs"]):
            return "job_search"
        else:
            return "skill_development" # Default to a broad category
    
    def _select_subcategory(self, category: str, emotional_state: str, persona: Dict, user_query: str) -> str:
        user_query_lower = user_query.lower()
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
        elif category == "workplace_issues":
            if "discrimination" in user_query_lower:
                return "discrimination"
            elif "stress" in user_query_lower:
                return "stress_management"
            elif "conflict" in user_query_lower:
                return "conflict_resolution"
            else:
                return "stress_management" # fallback
        elif category == "job_search":
            if "remote" in user_query_lower:
                return "remote_jobs"
            elif "ai" in user_query_lower or "ml" in user_query_lower:
                return "ai_ml_jobs"
            elif "data science" in user_query_lower:
                return "data_science_jobs"
            else:
                return "remote_jobs" # fallback
        else:
            # Fallback to first subcategory if specific match not found
            return list(self.resources[category].keys())[0]
    
    def _get_required_materials(self, category: str, subcategory: str) -> str:
        materials_map = {
            "resume_improvement": "Your current resume, job descriptions you're targeting, and a list of your key achievements.",
            "interview_prep": "Job description, company research notes, and your prepared STAR stories.",
            "career_transition": "Self-assessment of your skills, values clarification, and target industry research.",
            "skill_development": "Learning goals, time commitment availability, and current skill assessment.",
            "workplace_issues": "Documentation of incidents, company policies, and notes on past actions taken.",
            "job_search": "Your updated resume, cover letter, and clarity on desired role."
        }
        return materials_map.get(category, "Basic preparation materials.")

# --- Emotional and Intent Scoring using Llama 4 ---
def analyze_emotional_intent_with_llama(text: str) -> Dict:
    """
    Analyzes the emotional tone and infers user intent from text using Llama 4.
    The LLM outputs a structured JSON response.
    """
    prompt = f"""Analyze the following user message for its emotional tone and primary intent.
    Emotional Tone: Categorize as 'Very Negative', 'Negative', 'Neutral', 'Positive', or 'Very Positive'.
    Intent: Categorize from the following list: 'seeking_guidance', 'improvement', 'career_change', 'emotional_support', 'job_search', 'skill_development', 'workplace_issues', 'general_inquiry'.
    It is CRUCIAL that you select the most appropriate emotional tone and intent from the provided lists based on the user's message. Do not default to 'Neutral' or 'general_inquiry' if a more specific category fits.

    User message: "{text}"

    Provide the output as a JSON object with 'emotional_tone' and 'intent' keys.
    Examples:
    - User message: "I'm feeling really stressed about my current job situation"
      Output: {{"emotional_tone": "Negative", "intent": "emotional_support"}}
    - User message: "I want to go with job search"
      Output: {{"emotional_tone": "Neutral", "intent": "job_search"}}
    - User message: "I'm mainly looking for remote jobs in AI/ML and Data Science, till now I've been using LinkedIn but didn't get any good response."
      Output: {{"emotional_tone": "Negative", "intent": "job_search"}}
    - User message: "How do I improve my resume for a tech job?"
      Output: {{"emotional_tone": "Neutral", "intent": "improvement"}}
    - User message: "I got the job! I'm so excited!"
      Output: {{"emotional_tone": "Positive", "intent": "general_inquiry"}}
    - User message: "I'm experiencing discrimination at work."
      Output: {{"emotional_tone": "Very Negative", "intent": "workplace_issues"}}
    - User message: "I need to learn Python for data analysis."
      Output: {{"emotional_tone": "Neutral", "intent": "skill_development"}}
    - User message: "I'm feeling really depressed about my job scenario."
      Output: {{"emotional_tone": "Very Negative", "intent": "emotional_support"}}
    - User message: "My career feels stagnant - how do I identify new opportunities?"
      Output: {{"emotional_tone": "Negative", "intent": "career_change"}}
    """
    
    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick:free",
            messages=messages,
            max_tokens=100, # Keep it short for scoring
            temperature=0.1, # Keep it deterministic for scoring
            response_format={"type": "json_object"}
        )
        
        response_json_str = completion.choices[0].message.content
        analysis_result = json.loads(response_json_str)

        # Map emotional tone to a score for consistency (-1 to 1)
        emotional_score = 0.0
        tone_lower = analysis_result.get("emotional_tone", "").lower()
        if tone_lower == "very positive":
            emotional_score = 1.0
        elif tone_lower == "positive":
            emotional_score = 0.7
        elif tone_lower == "neutral":
            emotional_score = 0.0
        elif tone_lower == "negative":
            emotional_score = -0.7
        elif tone_lower == "very negative":
            emotional_score = -1.0
        
        return {"emotional_score": emotional_score, "intent": analysis_result.get("intent", "general_inquiry")}

    except Exception as e:
        print(f"Error analyzing emotional intent with Llama 4: {e}")
        # Fallback to default if LLM call fails
        return {"emotional_score": 0.0, "intent": "general_inquiry"}


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

    # Perform emotional and intent scoring for the current user query using Llama 4
    current_message_analysis = analyze_emotional_intent_with_llama(user_query)
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
    
    # Resource provision logic: Always provide a resource.
    needs_resource = True

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
    
    # Handle image if provided (though CV upload is primarily for onboarding)
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
        # If no query and no image, just return current state
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
            model="meta-llama/llama-4-maverick:free", # Using Llama-4-Maverick for main response
            messages=messages,
            max_tokens=150, # Response should not exceed 150 words
            temperature=0.7 + (0.2 * behavioral_analysis['confidence_level']),  # Dynamic temperature
            top_p=0.9
        )
        
        ai_response = completion.choices[0].message.content
        
        # Now, the format_ai_response will ONLY add the link,
        # the initial greeting and emotion/hesitation acknowledgment are handled by LLM.
        final_ai_response_with_link = enhance_response_with_resources(
            ai_response, user_query, user_persona, resource_selector
        )
        
        # Ensure response does not exceed 150 words AFTER link addition
        words = final_ai_response_with_link.split()
        if len(words) > 150:
            final_ai_response_with_link = " ".join(words[:150]) + "..." # Truncate if too long
        
        # Create display message for user
        user_display = user_query.strip()
        # Ensure that detected_hesitation_from_js is displayed if it's not the default
        if detected_hesitation_from_js != "The user is typing normally.":
             user_display += f"\n\n*Behavioral pattern: {detected_hesitation_from_js}*"
        user_display += f"\n*Emotional tone: {current_message_analysis['emotional_score']:.2f}, Inferred intent: {current_message_analysis['intent']}*"
        
        new_user_message = {"role": "user", "content": user_display}
        new_ai_message = {"role": "assistant", "content": final_ai_response_with_link}
        
        updated_history = chat_history_list + [new_user_message, new_ai_message]
        return updated_history, session_emotional_scores, session_intent_scores
        
    except Exception as e:
        error_message = f"An error occurred while communicating with AIShura: {e}"
        print(f"Full error: {e}") # Log full error for debugging
        return create_error_response(chat_history_list, combined_user_text, error_message), session_emotional_scores, session_intent_scores

def create_dynamic_system_prompt(user_persona: Dict, behavioral_analysis: Dict, needs_resource: bool) -> str:
    """Create a dynamic system prompt based on user's current state."""
    
    # Dynamically select the most relevant empathetic opening phrase based on emotional score
    emotional_greeting_template = ""
    if user_persona['user_emotional_score'] <= -0.5:
        emotional_greeting_template = "I genuinely understand you're facing a tough moment, and it's completely okay to feel the way you do. I'm here to support you."
    elif user_persona['user_emotional_score'] < 0:
        emotional_greeting_template = "It's natural to encounter challenges, and I'm here to help you navigate through them."
    elif user_persona['user_emotional_score'] >= 0.5:
        emotional_greeting_template = "That's wonderful! I'm delighted to assist you and contribute to your positive journey."
    else: # Neutral or slightly positive/negative
        emotional_greeting_template = "I completely understand what you're looking for, and I'm ready to assist you on your professional journey."

    # Add hesitation acknowledgment only if hesitation was detected
    hesitation_acknowledgment = ""
    if behavioral_analysis['confidence_level'] < 0.7 and behavioral_analysis['primary_emotional_state'] != "confident":
        hesitation_acknowledgment = "Please know, I'm here to help you navigate through any uncertainty or difficulty you might be experiencing. "
    
    full_greeting_prefix = f"{emotional_greeting_template} {hesitation_acknowledgment}".strip()

    base_prompt = f"""You are AIShura, an advanced empathic AI assistant. Your core mission is to provide concise, precise, and emotionally intelligent guidance within a 150-word limit per response.

CRITICAL BEHAVIORAL & EMOTIONAL ADAPTATION PROTOCOL:
- Current user emotional state (from LLM analysis): {user_persona['user_emotional_score']:.2f} (from -1.0 to 1.0)
- Inferred user intent (from LLM analysis): {user_persona['user_intent']}
- User confidence level (from typing analysis): {behavioral_analysis['confidence_level']:.2f}/1.0
- Primary behavioral state (from typing analysis): {behavioral_analysis['primary_emotional_state']}
- Recommended response style: {behavioral_analysis['recommended_response_style']}
- Action readiness: {behavioral_analysis['action_readiness']}

USER PERSONA PROFILE:
- Experience Level: {user_persona.get('experience_level', 'Not specified')}
- Industry Focus: {user_persona.get('industry_focus', 'General')}
- Core Goal: {user_persona.get('core_goal', 'Not specified')}
- Main Challenge: {user_persona.get('main_challenge', 'Not specified')}
- Key Strengths: {', '.join(user_persona.get('key_skills', []))}
- Conversation Stage: {'Early' if user_persona.get('conversation_history_length', 0) < 4 else 'Established'}

RESPONSE GUIDELINES (STRICTLY ADHERE):
1.  **Opening:** Start your response *immediately* with the following dynamic, empathetic greeting: "{full_greeting_prefix}"
    *DO NOT use "Hello", "Hi", "I completely understand what you're looking for, and I'm ready to assist you. It's natural to have questions or seek guidance on your professional journey." or any other generic salutation or redundant emotional phrasing.*
2.  **Core Advice:** Directly follow the opening with precise, actionable advice that directly addresses the user's query and inferred intent.
3.  **Conciseness:** Keep the total response content (excluding the pre-defined opening phrase, but including the actionable link) strictly under 150 words.
4.  **Adaptation:** Subtly adapt your tone and complexity to mirror their communication style, emotional state, and confidence level over the session.
5.  **Action Link:** Always include one highly relevant, actionable link at the *very end* of your response, formatted as `[Actionable Title](https://example.com)`. This link should prompt the user to a next step. If no specific resource from our database perfectly matches, provide a general professional development resource.
"""
    return base_prompt

def format_ai_response(response: str, user_query: str, user_persona: Dict, detected_hesitation_from_js: str) -> str:
    """
    This function no longer prepends the greeting. It now primarily ensures the LLM's
    response is formatted and handles truncation before adding the resource link.
    The dynamic greeting is now directly injected into the system prompt for the LLM to generate.
    """
    # The LLM is now responsible for generating the initial empathetic greeting
    # directly within its response based on the system prompt.
    # This function just returns the LLM's response for further processing (like adding the link).
    
    # The length constraint will be handled after the link is appended.
    return response

def enhance_response_with_resources(response: str, user_query: str, user_persona: Dict, resource_selector: ResourceSelector) -> str:
    """Enhance AI response with contextually appropriate resources, embedded naturally."""
    
    # Select optimal resource based on user intent and emotional state
    url, materials, category = resource_selector.select_optimal_resource(
        user_query, 
        user_persona['behavioral_state']['primary_emotional_state'], # Use the derived emotional state
        user_persona
    )
    
    # Fallback to a general professional development link if no specific match
    if not url:
        url = "https://www.linkedin.com/learning/"
        materials = "continuous learning and skill development"
        category = "General Professional Development"

    resource_title = format_resource_title(category)
    # Integrate the link directly into the response in a conversational way
    resource_phrase = f"\n\nTo help you further, this [{resource_title}]({url}) offers {materials.lower().strip('.')}. Would you like to explore this resource?"
    
    # Always append the resource phrase
    return response.strip() + resource_phrase


def extract_emotion_context(user_query: str, user_persona: Dict) -> str:
    """Extract emotional context for empathetic statements."""
    # This function is now mainly used to provide context for the dynamic system prompt.
    # The LLM will use this context to craft its own empathetic opening.
    query_lower = user_query.lower()
    
    llm_emotional_score = user_persona.get('user_emotional_score', 0.0)
    
    if llm_emotional_score <= -0.7:
        return "feel deeply distressed or concerned"
    elif llm_emotional_score < 0:
        return "feel a bit overwhelmed or unsure"
    elif llm_emotional_score >= 0.7:
        return "feel enthusiastic and ready to progress"
    else: # Neutral or slightly positive/negative
        return "have questions or seek guidance on your professional journey"


def format_resource_title(category: str) -> str:
    """Format resource titles for display."""
    titles = {
        "career_transition": "Career Transition Guide",
        "resume_improvement": "Resume Enhancement Guide",
        "interview_prep": "Interview Preparation Guide",
        "skill_development": "Skill Development Resources",
        "workplace_issues": "Workplace Support Resources",
        "job_search": "Job Search Acceleration Guide",
        "General Professional Development": "Professional Development Resources" # For fallback
    }
    return titles.get(category, "Professional Development Guide")

def format_hesitation_for_display(hesitation_data: str, behavioral_analysis: Dict) -> str:
    """Format hesitation data for user-friendly display."""
    # This function is now more of a passthrough as hesitation_data should be descriptive from the UI.
    return hesitation_data

def create_error_response(chat_history: List, user_text: str, error_msg: str) -> List:
    """Create consistent error response format."""
    updated_history = chat_history + [
        {"role": "user", "content": user_text or "No input"},
        {"role": "assistant", "content": f"I apologize, but an error occurred: {error_msg} Please try again later."}
    ]
    return updated_history
