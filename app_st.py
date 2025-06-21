import streamlit as st
import json
import time
import pandas as pd
import os
# Assume the AI logic (HesitationAnalyzer, ResourceSelector, process_aisura_query etc.)
# is imported from the app_logic.py file you've created
from app_logic import (
    HesitationAnalyzer, ResourceSelector, analyze_cv_content,
    analyze_emotional_intent_with_llama, process_aisura_query, # Use the Llama-based analysis
    format_hesitation_for_display
)

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="AIShura: Empathic AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed" # Start collapsed for focus on main content
)

# Create a temporary directory for CV uploads if it doesn't exist
os.makedirs("./temp_cv", exist_ok=True)


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
        color: #e0e0e0;
    }
    .stChatMessage.assistant {
        background-color: #3e285e; /* Darker purple for assistant */
        border-top-right-radius: 5px;
        color: #e0e0e0;
    }
    .stChatMessage .stMarkdown {
        color: #e0e0e0;
    }
    /* Links within markdown */
    .stChatMessage a {
        color: #90ee90; /* Light green for links */
        text-decoration: none;
    }
    .stChatMessage a:hover {
        text-decoration: underline;
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
    .example-query-btn {
        background-color: #1e1e3a;
        border-left: 3px solid #764ba2;
        padding: 8px 12px;
        margin: 5px 0;
        border-radius: 5px;
        font-size: 0.9em;
        cursor: pointer;
        transition: background-color 0.2s ease;
        color: #e0e0e0; /* Ensure text is visible */
        border: none; /* Make it look more like a clickable div than a button */
        width: 100%;
        text-align: left;
    }
    .example-query-btn:hover {
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
# Initialize user_query_input for the main text area
if 'user_query_input' not in st.session_state:
    st.session_state.user_query_input = ""
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
        key="mood_desc_input",
        value=st.session_state.onboarding_data["mood_desc"] # Pre-fill if already entered
    )
    
    core_goal_input = st.text_area(
        label="What brings you to AIShura today? What's your primary objective?",
        placeholder="e.g., 'Transform my resume for tech roles' or 'Navigate a complete career pivot'",
        height=100,
        key="core_goal_input",
        value=st.session_state.onboarding_data["core_goal"] # Pre-fill if already entered
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
            key="challenge_input",
            value=st.session_state.onboarding_data["challenge"] # Pre-fill
        )
        
        outcome_input = st.text_area(
            label="What specific outcome would make this conversation successful for you?",
            placeholder="e.g., 'A clear 30-day action plan' or 'Confidence in my career direction'",
            height=150,
            key="outcome_input",
            value=st.session_state.onboarding_data["outcome"] # Pre-fill
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
        cv_path = None
        if cv_upload:
            try:
                # Create the directory if it doesn't exist
                os.makedirs("./temp_cv", exist_ok=True)

                # Save temporary file for analysis
                file_extension = os.path.splitext(cv_upload.name)[1]
                temp_cv_filename = f"uploaded_cv_{int(time.time())}{file_extension}"
                cv_path = os.path.join("./temp_cv", temp_cv_filename)
                
                with open(cv_path, "wb") as f:
                    f.write(cv_upload.getbuffer())
                
                with st.spinner("Analyzing your CV..."):
                    # The analyze_cv_content function is currently a simulation
                    # In a real app, this would involve OCR/NLP services
                    cv_analysis = analyze_cv_content(cv_path) 
                    st.session_state.onboarding_data["cv_analysis"] = json.dumps(cv_analysis)
                st.success("CV uploaded and analyzed successfully! Proceed to next step.")
                # Optional: Clean up the temporary file after analysis if no longer needed
                # os.remove(cv_path) # Uncomment if you want to delete the file immediately
            except Exception as e:
                st.error(f"Failed to process CV: {e}. Please try again or upload a different file type.")
                print(f"CV Upload Error: {e}") # For internal debugging

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
        key="main_text_input",
        value=st.session_state.user_query_input # Bind to session state for example clicks
    )

    # Simplified hesitation display: will be updated based on submit action
    hesitation_display_text = st.session_state.detected_hesitation
    st.markdown(
        f'<div class="hesitation-indicator">🧠 AIShura\'s Behavioral Analysis: {hesitation_display_text}</div>',
        unsafe_allow_html=True
    )

    submit_button = st.button("Get AIShura's Insight", type="primary", key="submit_button")

    # Function to handle example clicks
    def set_example_query(query_text):
        st.session_state.user_query_input = query_text

    if submit_button and user_query:
        # Simulate hesitation data for demonstration.
        # In a real deployed app with st.components.v1.html, you'd get this from JS.
        # For a pitch, you can cycle through some pre-defined states or explain it's real-time.
        simulated_hesitation = "The user is typing normally."
        if "stressed" in user_query.lower() or "anxious" in user_query.lower() or "struggle" in user_query.lower():
             simulated_hesitation = "Thoughtful consideration detected - anxious state (low confidence)"
        elif len(user_query.split()) < 5: # Very short queries
            simulated_hesitation = "Concise input - may indicate urgency or high confidence (medium confidence)"
        elif len(user_query) > 100: # Long queries
            simulated_hesitation = "Detailed input - contemplative state (high confidence)"
        
        st.session_state.detected_hesitation = simulated_hesitation # Update for next display

        current_cv_path = None # Assuming no image upload in main chat after onboarding

        with st.spinner("AIShura is adapting and generating your personalized insight..."):
            updated_chat_history, updated_emotional_scores, updated_intent_scores = process_aisura_query(
                image_path=current_cv_path,
                user_query=user_query,
                onboarding_mood_desc=st.session_state.onboarding_data["mood_desc"],
                onboarding_core_goal=st.session_state.onboarding_data["core_goal"],
                onboarding_challenge=st.session_state.onboarding_data["challenge"],
                onboarding_outcome=st.session_state.onboarding_data["outcome"],
                onboarding_cv_analysis=st.session_state.onboarding_data["cv_analysis"],
                detected_hesitation_from_js=st.session_state.detected_hesitation, # Pass the simulated data
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
                # Extract confidence directly from behavioral_analysis returned by process_aisura_query if possible
                # Or re-calculate for display based on the last behavioral analysis used for response
                last_behavioral_analysis = HesitationAnalyzer().analyze_hesitation_pattern(
                    st.session_state.detected_hesitation, user_query, st.session_state.chat_history # Re-run for confidence
                )
                
                st.session_state.user_behavior_data = pd.concat([
                    st.session_state.user_behavior_data,
                    pd.DataFrame([{
                        'Turn': int(turn_num),
                        'Emotional Score': updated_emotional_scores[-1],
                        'Confidence': last_behavioral_analysis['confidence_level'] * 100 # Convert to percentage
                    }])
                ], ignore_index=True)
            
        st.session_state.user_query_input = "" # Clear input after submission
        st.rerun() # Rerun to update chat and clear input

    st.markdown("---")
    st.markdown("### 💡 Try these sophisticated queries")
    examples = [
        "I'm struggling to make my resume stand out in the tech industry",
        "How do I transition from marketing to data science?",
        "I have interview anxiety - what specific techniques can help?",
        "My career feels stagnant - how do I identify new opportunities?"
    ]
    # Use st.button for examples, which is more robust for triggering actions
    # When clicked, set the value of the main_text_input in session state
    # and then rerun the app.
    for i, example in enumerate(examples):
        # Using a unique key for each button is crucial
        st.button(example, key=f"example_btn_{i}", help="Click to pre-fill the chat input with this query.", on_click=set_example_query, args=(example,), use_container_width=True)


    # --- Dashboard (Sidebar or expandable section) ---
    with st.sidebar:
        st.markdown("## 📊 AIShura Intelligence Dashboard")
        st.markdown("---")

        st.markdown("### User Emotional Trend")
        if not st.session_state.user_behavior_data.empty:
            # Create a Plotly chart for better aesthetics and interactivity
            import plotly.express as px
            fig_emo = px.line(
                st.session_state.user_behavior_data,
                x='Turn',
                y='Emotional Score',
                title='Emotional Score Over Time',
                labels={'Emotional Score': 'Emotional Score (-1 to 1)'},
                color_discrete_sequence=px.colors.qualitative.Pastel # Softer color palette
            )
            fig_emo.update_layout(xaxis_title="Conversation Turn", yaxis_title="Emotional Score")
            st.plotly_chart(fig_emo, use_container_width=True)
            
            st.markdown("### User Confidence Over Time")
            # Convert Confidence to numeric, handling potential errors
            st.session_state.user_behavior_data['Confidence_Numeric'] = pd.to_numeric(
                st.session_state.user_behavior_data['Confidence'], errors='coerce'
            )
            fig_conf = px.line(
                st.session_state.user_behavior_data,
                x='Turn',
                y='Confidence_Numeric',
                title='Confidence Level Over Time',
                labels={'Confidence_Numeric': 'Confidence (%)'},
                color_discrete_sequence=px.colors.qualitative.Set2 # Another color palette
            )
            fig_conf.update_layout(xaxis_title="Conversation Turn", yaxis_title="Confidence (%)")
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.info("Start chatting to see behavioral analytics!")

        st.markdown("### Inferred User Intents")
        if st.session_state.session_intent_scores:
            intent_counts = pd.Series(st.session_state.session_intent_scores).value_counts().reset_index()
            intent_counts.columns = ['Intent', 'Count']
            fig_intent = px.bar(
                intent_counts,
                x='Intent',
                y='Count',
                title='Distribution of Inferred Intents',
                color='Intent', # Color bars by intent
                color_discrete_sequence=px.colors.qualitative.D3 # Distinct colors
            )
            fig_intent.update_layout(xaxis_title="User Intent", yaxis_title="Number of Occurrences")
            st.plotly_chart(fig_intent, use_container_width=True)
        else:
            st.info("Intents will appear here as you chat.")

        st.markdown("### Current User Persona")
        # Display onboarding data more cleanly
        st.json(st.session_state.onboarding_data)
