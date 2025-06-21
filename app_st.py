import streamlit as st
import json
import time
import pandas as pd # For charts
# Assume the AI logic (HesitationAnalyzer, ResourceSelector, process_aisura_query etc.)
# is imported from the app_logic.py file you've created (the first code block above)
from app_logic import (
    HesitationAnalyzer, ResourceSelector, analyze_cv_content,
    analyze_emotional_intent, process_aisura_query,
    format_hesitation_for_display # You'll need to expose these
)

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="AIShura: Empathic AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded" # Or collapsed initially for focus
)

# Apply custom CSS for futuristic look
st.markdown("""
<style>
    /* General body and container styling */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    .stSidebar {
        background-color: #0f0f1a;
        color: #e0e0e0;
    }
    .st-emotion-cache-vk33gh { /* Target specific element for main content padding */
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .st-emotion-cache-16txtv4 { /* Target element for chat input and submit button arrangement */
        gap: 1rem;
    }

    /* Chatbot styling */
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 10px;
        padding: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }
    .stChatMessage.user {
        background-color: #283a5e; /* Darker blue for user */
        border-top-left-radius: 5px;
    }
    .stChatMessage.assistant {
        background-color: #3e285e; /* Darker purple for assistant */
        border-top-right-radius: 5px;
    }
    .stChatMessage .stMarkdown {
        color: #e0e0e0;
    }

    /* Input elements */
    .stTextInput label, .stTextArea label, .stSelectbox label, .stFileUpload label {
        color: #a0a0ff; /* Lighter blue for labels */
        font-weight: bold;
    }
    .stTextInput div div input, .stTextArea textarea, .stSelectbox div div {
        background-color: #0f0f1a;
        color: #e0e0e0;
        border: 1px solid #4a4a6e;
        border-radius: 8px;
        padding: 10px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #667eea;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #764ba2;
    }
    .stButton.primary > button { /* For primary button variant */
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
    }
    .stButton.primary > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }

    /* Markdown styling */
    h1, h2, h3, h4 {
        color: #8c9eff; /* Lighter purple/blue for headers */
    }
    .onboarding-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.4);
    }
    .hesitation-indicator {
        background-color: #202c45;
        border-left: 5px solid #0ea5e9;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        font-style: italic;
        color: #c0c0f0;
        font-size: 0.9em;
    }
    .example-query {
        background-color: #1e1e3a;
        border-left: 3px solid #764ba2;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 5px;
        font-size: 0.9em;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    .example-query:hover {
        background-color: #2b2b4d;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
if 'onboarding_step' not in st.session_state:
    st.session_state.onboarding_step = 1
if 'onboarding_data' not in st.session_state:
    st.session_state.onboarding_data = {
        "mood_desc": "", "core_goal": "", "challenge": "",
        "outcome": "", "cv_analysis": "{}"
    }
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'detected_hesitation' not in st.session_state:
    st.session_state.detected_hesitation = "The user is typing normally."
if 'session_emotional_scores' not in st.session_state:
    st.session_state.session_emotional_scores = []
if 'session_intent_scores' not in st.session_state:
    st.session_state.session_intent_scores = []
if 'user_behavior_data' not in st.session_state:
    st.session_state.user_behavior_data = pd.DataFrame(columns=['Turn', 'Emotional Score', 'Confidence'])


# --- Header ---
st.markdown(
    """
    <div class="onboarding-section">
        <h1>🧠 AIShura: Advanced Empathic AI Assistant</h1>
        <h2><i>Detects hesitation. Adapts in real-time. Delivers precisely what you need.</i></h2>
        <p><b>The world's first AI that understands not just what you say, but how you feel while saying it.</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Onboarding Flow ---
if st.session_state.onboarding_step == 1:
    st.markdown("### 🎯 Step 1: Emotional Landscape Assessment")
    st.markdown("*AIShura adapts its entire response strategy based on your current emotional state*")
    
    mood_desc_input = st.text_area(
        label="What's your current emotional state regarding your professional journey?",
        placeholder="e.g., 'I'm feeling overwhelmed by career options' or 'Excited but nervous about job searching'",
        height=100,
        key="mood_desc_input"
    )
    
    core_goal_input = st.text_area(
        label="What brings you to AIShura today? What's your primary objective?",
        placeholder="e.g., 'Transform my resume for tech roles' or 'Navigate a complete career pivot'",
        height=100,
        key="core_goal_input"
    )
    
    if st.button("Continue to Deep Personalization →", type="primary", key="onboard_button_1"):
        if mood_desc_input and core_goal_input:
            st.session_state.onboarding_data["mood_desc"] = mood_desc_input
            st.session_state.onboarding_data["core_goal"] = core_goal_input
            st.session_state.onboarding_step = 2
            st.rerun()
        else:
            st.warning("Please fill in both fields to continue.")

elif st.session_state.onboarding_step == 2:
    st.markdown("### 🎯 Step 2: Challenge Identification & Outcome Visualization")
    st.markdown("*AIShura calibrates its response precision based on your specific needs*")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        challenge_input = st.text_area(
            label="What specific challenge is blocking your progress right now?",
            placeholder="e.g., 'Can't articulate my value proposition' or 'Struggling with career direction'",
            height=150,
            key="challenge_input"
        )
        
        outcome_input = st.text_area(
            label="What specific outcome would make this conversation successful for you?",
            placeholder="e.g., 'A clear 30-day action plan' or 'Confidence in my career direction'",
            height=150,
            key="outcome_input"
        )
    
    with col2:
        cv_upload = st.file_uploader(
            label="📄 Upload Your Resume/CV (Optional but Recommended)",
            type=["pdf", "docx", "png", "jpg", "jpeg"],
            key="cv_upload"
        )
        st.markdown(
            """
            *Uploading your CV enables AIShura to build a complete professional persona 
            and provide hyper-targeted guidance throughout your conversation.*
            """
        )
        if cv_upload:
            # Save temporary file for analysis
            with open(os.path.join("./temp_cv", cv_upload.name), "wb") as f:
                f.write(cv_upload.getbuffer())
            cv_path = os.path.join("./temp_cv", cv_upload.name)
            cv_analysis = analyze_cv_content(cv_path) # Call your analysis function
            st.session_state.onboarding_data["cv_analysis"] = json.dumps(cv_analysis)
            st.success("CV uploaded and analyzed!")

    if st.button("Launch My AIShura Experience ✨", type="primary", key="onboard_button_2"):
        if challenge_input and outcome_input:
            st.session_state.onboarding_data["challenge"] = challenge_input
            st.session_state.onboarding_data["outcome"] = outcome_input
            st.session_state.onboarding_step = 3
            st.rerun()
        else:
            st.warning("Please fill in challenge and outcome to launch.")

# --- Main Chat Interface ---
elif st.session_state.onboarding_step == 3:
    st.markdown("### 💬 Your AIShura Conversation")
    st.markdown("*Watch as AIShura adapts its responses based on your typing patterns and emotional state*")

    # Chatbot display
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f'<div class="stChatMessage user">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="stChatMessage assistant">🧠 {message["content"]}</div>', unsafe_allow_html=True)

    # Input and controls
    user_query = st.text_area(
        label="Share your thoughts, questions, or challenges",
        placeholder="e.g., 'How do I position my skills for senior roles?' or 'I'm struggling with interview confidence'",
        height=100,
        key="main_text_input"
    )

    # Simplified hesitation display (requires JS for real-time)
    hesitation_display_placeholder = st.empty()
    hesitation_display_placeholder.markdown(
        f'<div class="hesitation-indicator">{st.session_state.detected_hesitation}</div>',
        unsafe_allow_html=True
    )

    # Note: Real-time hesitation from JS to Streamlit is complex.
    # For a pitch, you could either:
    # 1. Simulate it by updating `st.session_state.detected_hesitation` on a timer (not true user input)
    # 2. Use `st.components.v1.html` for a true JS integration (advanced)
    # For this example, we will just use a static placeholder or manual update.

    submit_button = st.button("Get AIShura's Insight", type="primary", key="submit_button")

    if submit_button and user_query:
        # Simulate hesitation data for demonstration if not coming from JS
        # In a real app, this would be passed from the JavaScript component
        # For pitch, you could cycle through some pre-defined states here
        simulated_hesitation = "Confident expression - clear communication flow"
        if len(user_query) < 20:
             simulated_hesitation = "Thoughtful consideration detected - neutral state"
        elif "anxious" in user_query.lower() or "struggle" in user_query.lower():
             simulated_hesitation = "High anxiety - extensive self-correction and hesitation"
        
        st.session_state.detected_hesitation = simulated_hesitation # Update for display

        current_cv_path = None # Assuming no image upload in main chat for simplicity

        with st.spinner("AIShura is adapting and generating your personalized insight..."):
            updated_chat_history, updated_emotional_scores, updated_intent_scores = process_aisura_query(
                image_path=current_cv_path,
                user_query=user_query,
                onboarding_mood_desc=st.session_state.onboarding_data["mood_desc"],
                onboarding_core_goal=st.session_state.onboarding_data["core_goal"],
                onboarding_challenge=st.session_state.onboarding_data["challenge"],
                onboarding_outcome=st.session_state.onboarding_data["outcome"],
                onboarding_cv_analysis=st.session_state.onboarding_data["cv_analysis"],
                detected_hesitation_from_js=st.session_state.detected_hesitation,
                chat_history_list=st.session_state.chat_history,
                session_emotional_scores=st.session_state.session_emotional_scores,
                session_intent_scores=st.session_state.session_intent_scores
            )
            
            st.session_state.chat_history = updated_chat_history
            st.session_state.session_emotional_scores = updated_emotional_scores
            st.session_state.session_intent_scores = updated_intent_scores

            # Update user behavior data for dashboard
            turn_num = len(st.session_state.chat_history) / 2
            if updated_emotional_scores: # Ensure list is not empty
                st.session_state.user_behavior_data = pd.concat([
                    st.session_state.user_behavior_data,
                    pd.DataFrame([{
                        'Turn': int(turn_num),
                        'Emotional Score': updated_emotional_scores[-1],
                        'Confidence': updated_chat_history[-1]['content'].split('Confidence: ')[1].split('%')[0] if 'Confidence: ' in updated_chat_history[-1]['content'] else 0.5 # Extract from display text
                    }])
                ], ignore_index=True)
            
        st.rerun() # Rerun to clear input and update chat

    st.markdown("---")
    st.markdown("### 💡 Try these sophisticated queries")
    examples = [
        "I'm struggling to make my resume stand out in the tech industry",
        "How do I transition from marketing to data science?",
        "I have interview anxiety - what specific techniques can help?",
        "My career feels stagnant - how do I identify new opportunities?"
    ]
    for example in examples:
        if st.markdown(f'<div class="example-query">{example}</div>', unsafe_allow_html=True):
            st.session_state.main_text_input = example # Set text input
            st.rerun()

    # --- Dashboard (Sidebar or expandable section) ---
    with st.sidebar:
        st.markdown("## 📊 AIShura Intelligence Dashboard")
        st.markdown("---")

        st.markdown("### User Emotional Trend")
        if st.session_state.user_behavior_data is not None and not st.session_state.user_behavior_data.empty:
            st.line_chart(st.session_state.user_behavior_data.set_index('Turn')['Emotional Score'])
            
            st.markdown("### User Confidence Over Time")
            # Convert Confidence to numeric, handling potential errors
            st.session_state.user_behavior_data['Confidence_Numeric'] = pd.to_numeric(
                st.session_state.user_behavior_data['Confidence'], errors='coerce'
            )
            st.line_chart(st.session_state.user_behavior_data.set_index('Turn')['Confidence_Numeric'])
        else:
            st.info("Start chatting to see behavioral analytics!")

        st.markdown("### Inferred User Intents")
        if st.session_state.session_intent_scores:
            intent_counts = pd.Series(st.session_state.session_intent_scores).value_counts()
            st.bar_chart(intent_counts)
        else:
            st.info("Intents will appear here as you chat.")

        st.markdown("### Current User Persona")
        st.json(st.session_state.onboarding_data) # Display raw onboarding data for simplicity
