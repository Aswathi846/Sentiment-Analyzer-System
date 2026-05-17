import sys
import os
import logging
import uuid
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import shap
import streamlit.components.v1 as components

# 1. Setup Path & Environment
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_engine import SentimentEngine
from src.data_processor import ReviewInput, TextCleaner
from src.agent import SentimentAgent 

# 2. Configure Logging
logging.basicConfig(level=logging.INFO)

# 3. Page Configuration 
st.set_page_config(
    page_title="Sentiment Analyzer System", 
    layout="wide", 
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

# 4. Premium SaaS UI Style Injector
st.markdown("""
    <style>
        /* Base page application layout */
        .main {
            background-color: #f8fafc !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        
        /* Force high-visibility typography contrasts */
        h1, h2, h3, h4, p, span, label, .stMarkdown {
            color: #0f172a !important;
        }
        
        /* Remove excessive top padding from the main Streamlit page container */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Eliminate native block-gap padding elements globally */
        div[data-testid="stVerticalBlock"] > div {
            margin-bottom: 0px !important;
            padding-bottom: 0px !important;
            margin-top: 0px !important;
        }
        
        /* Input Form Elements customization definitions */
        div[data-testid="stTextArea"] textarea {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
            color: #0f172a !important;
            font-size: 15px !important;
        }
        div[data-testid="stTextArea"] textarea:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 1px #2563eb !important;
        }

        /* Group Pill Labeling Framework styling components */
        .section-pill-label {
            background-color: #eff6ff;
            color: #2563eb !important;
            font-size: 11px !important;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 6px;
            display: inline-block;
            margin-top: 12px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* System status indicator badge banner style */
        .status-pill-green {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #166534 !important;
            font-size: 13px;
            font-weight: 500;
            padding: 8px 16px;
            border-radius: 30px;
            display: inline-block;
            margin-bottom: 12px;
        }
        
        /* App Functional Navigation Action Row Styling Reset */
        div.stButton > button {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            color: #475569 !important;
            border-radius: 9999px !important;
            padding: 8px 22px !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease;
        }
        div.stButton > button:hover {
            background-color: #f1f5f9 !important;
            color: #0f172a !important;
            border-color: #94a3b8 !important;
        }
        
        /* Master Engine Blue Action processing execution button */
        div.stButton > button[kind="primary"] {
            background-color: #2563eb !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 9999px !important;
            padding: 10px 32px !important;
            font-size: 15px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
            width: 100% !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #1d4ed8 !important;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3) !important;
        }
        
        /* Classification display components formatting panels */
        .verdict-box-pos {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #166534 !important;
            font-size: 24px;
            font-weight: 700;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        .verdict-box-neg {
            background-color: #fef2f2;
            border: 1px solid #fee2e2;
            color: #991b1b !important;
            font-size: 24px;
            font-weight: 700;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# 5. Resource Loading
@st.cache_resource
def load_resources():
    engine = SentimentEngine()
    engine.load_inference_pipeline()
    try:
        agent = SentimentAgent()
    except Exception as e:
        st.error(f"Error initializing Agent: {e}")
        agent = None
    return engine, agent

engine, agent = load_resources()
cleaner = TextCleaner()

# 6. SHAP Interactive Rendering Block Wrapper Setup
def st_shap(plot_obj, height=220):
    try:
        unique_id = f"shap_{uuid.uuid4().hex[:8]}"
        plot_html = plot_obj.html() if hasattr(plot_obj, "html") else str(plot_obj)
        final_html = f"""
            <div id="{unique_id}" style="background-color: #ffffff; padding: 12px; border: 1px solid #e2e8f0; border-radius: 8px;">
                <script>{shap.getjs()}</script>
                {plot_html}
            </div>
        """
        components.html(final_html, height=height + 24, scrolling=True)
    except Exception as e:
        st.error(f"Visualization error: {e}")

def render_shap_heatmap(text_input, pipeline):
    try:
        explainer = shap.Explainer(pipeline)
        shap_values = explainer([text_input])
        plot_obj = shap.plots.text(shap_values[0], display=False)
        st_shap(plot_obj)
    except Exception as e:
        st.error(f"XAI Heatmap Error: {str(e)}")

# 7. Session State Management Configurations
if 'user_text' not in st.session_state:
    st.session_state.user_text = ""
if 'is_analyzing' not in st.session_state:
    st.session_state.is_analyzing = False

# Application Brand Main Header Area Setup
st.markdown("<h2 style='font-weight: 700; margin-top: 0px; margin-bottom: 20px;'>🧠 Sentiment Analyzer System</h2>", unsafe_allow_html=True)

# =========================================================================
# SYSTEM CONTROL WORKSPACE - SIDE-BY-SIDE SPLIT EXECUTION DECK
# =========================================================================
col_left_inputs, col_right_outputs = st.columns([1, 1.2], gap="large")

with col_left_inputs:
    # Module Segment 1: Prompt Input Matrix Block
    st.markdown('<span class="section-pill-label" style="margin-top: 0px;">Prompt / Input Expression</span>', unsafe_allow_html=True)
    user_input = st.text_area(
        "hidden_label",
        value=st.session_state.user_text, 
        height=130, 
        placeholder="Enter textual expressions to process or pick a quick option below...", 
        label_visibility="collapsed"
    )
    
    # Module Segment 2: Preset Testing Scenario Select Box
    st.markdown('<span class="section-pill-label">Quick Test Scenarios</span>', unsafe_allow_html=True)
    selected_preset = st.selectbox(
        "hidden_label",
        ["Select an automated testing phrase variant...", "This movie is fire 🔥", "The service was mid 💀", "Absolute banger of a track."],
        label_visibility="collapsed"
    )
    if selected_preset != "Select an automated testing phrase variant..." and selected_preset != st.session_state.user_text:
        st.session_state.user_text = selected_preset
        st.rerun()
        
    # Module Segment 3: Multimodal Reference Media Input Card
    st.markdown('<span class="section-pill-label">Contextual Visual Layer</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("hidden_label", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)
        
    # Module Segment 4: System Operational Metrics Diagnostics Panel 
    st.markdown('<span class="section-pill-label">System Hardware Pipeline</span>', unsafe_allow_html=True)
    st.markdown("""
        <div style="background-color: #ffffff; border: 1px solid #e2e8f0; padding: 12px 16px; border-radius: 8px; font-size: 13px; color: #475569;">
            <b>Neural Model Stack:</b> DistilBERT Core (Local)<br>
            <b>Reasoning Agent Architecture:</b> Gemini 1.5 Flash (Sync)
        </div>
    """, unsafe_allow_html=True)
    
    # Bottom Layout Action Triggers Panel Container Row
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns([1, 1.8])
    with btn_col1:
        if st.button("Reset Canvas"):
            st.session_state.user_text = ""
            st.session_state.is_analyzing = False
            st.rerun()
            
    with btn_col2:
        # Changes button copy to an explicit loading sign instantly upon click
        button_label = "Analyzing... ⚡" if st.session_state.is_analyzing else "Generate Analysis ✨"
        analyze_triggered = st.button(button_label, type="primary", disabled=st.session_state.is_analyzing)
        
        if analyze_triggered:
            st.session_state.is_analyzing = True
            st.rerun()

with col_right_outputs:
    if st.session_state.is_analyzing:
        if not user_input.strip():
            st.warning("Please enter a valid text prompt in the interaction deck before generating.")
            st.session_state.is_analyzing = False
        else:
            # 1. Fast Inference Processing Step
            clean_text = cleaner.sanitize(user_input)
            prediction = engine.predict_batch([clean_text])[0]
            reasoning = agent.explain_sentiment(user_input, prediction['label'], uploaded_file) if agent else "Agent runtime offline."
            
            # Active Status Generation Indicator Pill Block Banner
            st.markdown('<div class="status-pill-green" style="margin-top: 0px;">✦ Sentiment Diagnostics Generated</div>', unsafe_allow_html=True)
            
            # STAGE 1: Classification Verdict & Confidence Metrics (Loads immediately)
            st.markdown('<span class="section-pill-label">Classification Output</span>', unsafe_allow_html=True)
            if prediction["label"] == "POSITIVE":
                st.markdown('<div class="verdict-box-pos">POSITIVE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-box-neg">NEGATIVE</div>', unsafe_allow_html=True)
                
            # Confidence Certainty Bar metrics outputs block
            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            st.metric("Neural Confidence Probability Certainty Index", f"{prediction['score']*100:.2f}%")
            st.progress(prediction["score"])
            
            # STAGE 2: Targeted Loading Spinner solely for the SHAP Map calculations
            st.markdown('<span class="section-pill-label">Feature Attribution Map (SHAP)</span>', unsafe_allow_html=True)
            with st.spinner("Calculating SHAP feature attribution matrices (this may take a few seconds)..."):
                render_shap_heatmap(clean_text, engine.pipeline)
            
            # STAGE 3: Agent Reasoning explanation output
            st.markdown('<span class="section-pill-label">Agentic Reasoning Matrix Explanations</span>', unsafe_allow_html=True)
            st.markdown(f"""
                <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; font-size: 14.5px; line-height: 1.6; color: #334155;">
                    {reasoning}
                </div>
            """, unsafe_allow_html=True)
            
            # Reset latch state now that evaluation loops are completely populated
            st.session_state.is_analyzing = False
            
    else:
        # Ground State Empty Placeholder Box structure definition (Zero ghost components margins)
        st.markdown('<div class="status-pill-green" style="margin-color: 0px; background-color: #f1f5f9; border-color: #cbd5e1; color: #475569 !important;">○ Pipeline Stream Idle</div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="border: 2px dashed #cbd5e1; border-radius: 12px; min-height: 520px; display: flex; flex-direction: column; justify-content: center; align-items: center; background-color: #ffffff; padding: 30px;">
                <p style="color: #64748b !important; font-size: 14.5px; text-align: center; font-weight: 500; line-height: 1.5;">
                    Configure model prompt criteria and select <br>
                    <b>"Generate Analysis ✨"</b> to construct neural token metrics.
                </p>
            </div>
        """, unsafe_allow_html=True)