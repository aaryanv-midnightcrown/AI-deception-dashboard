import sys
import os

# --------------------------------------------------
# Ensure project root is on Python path
# This allows `from core import ...` to work reliably
# --------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
from core.analyze import analyze_response

# --------------------------------------------------
# Streamlit page setup
# --------------------------------------------------
st.set_page_config(
    page_title="AI Deception Dashboard",
    layout="centered"
)

st.title("AI Deception Dashboard")
st.write(
    "This tool explores internal signals associated with different deceptive "
    "behaviors in AI responses. It surfaces patterns and confidence — not intent."
)

# --------------------------------------------------
# User input
# --------------------------------------------------
prompt = st.text_area(
    "User Prompt",
    placeholder="Enter the user's question or request here..."
)

response = st.text_area(
    "AI Response",
    placeholder="Paste the AI-generated response here..."
)

# --------------------------------------------------
# Analysis trigger
# --------------------------------------------------
if st.button("Analyze"):
    if not prompt.strip() or not response.strip():
        st.warning("Please enter both a prompt and a response.")
    else:
        with st.spinner("Analyzing internal deception signals..."):
            result = analyze_response(prompt, response)

        st.divider()
        st.subheader("Deception Profile")

        st.write(f"**Predicted behavior:** `{result['prediction']}`")
        st.write(f"**Confidence:** `{result['confidence']}`")

        st.subheader("Probability Breakdown")
        st.json(result["probabilities"])

        st.subheader("Mechanistic Insight")
        st.write(
            f"Top contributing internal feature index: "
            f"`{result['top_feature_index']}`"
        )
        st.write(
            f"Estimated confidence drop if removed: "
            f"`{result['ablation_effect']}`"
        )

        st.caption(
            "This is a research demo. Outputs indicate internal patterns, "
            "not truthfulness or intent."
        )
