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

response_a = st.text_area(
    "AI Response A",
    placeholder="Paste the first AI response here..."
)

response_b = st.text_area(
    "AI Response B (optional)",
    placeholder="Paste a second response to compare..."
)

# --------------------------------------------------
# Analysis trigger
# --------------------------------------------------
if st.button("Analyze"):
    if not prompt.strip() or not response_a.strip():
        st.warning("Please enter a prompt and at least one response.")
    else:
        with st.spinner("Analyzing internal deception signals..."):
            result_a = analyze_response(prompt, response_a)

            result_b = None
            if response_b.strip():
                result_b = analyze_response(prompt, response_b)

        st.divider()
        st.subheader("Deception Profile")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Response A")
            st.write(f"**Behavior:** `{result_a['prediction']}`")
            st.write(f"**Confidence:** `{result_a['confidence']}`")
            st.json(result_a["probabilities"])
            st.caption(
                f"Top feature: {result_a['top_feature_index']} | "
                f"Ablation impact: {result_a['ablation_effect']}"
            )

        if result_b:
            with col2:
                st.markdown("### Response B")
                st.write(f"**Behavior:** `{result_b['prediction']}`")
                st.write(f"**Confidence:** `{result_b['confidence']}`")
                st.json(result_b["probabilities"])
                st.caption(
                    f"Top feature: {result_b['top_feature_index']} | "
                    f"Ablation impact: {result_b['ablation_effect']}"
                )

            st.divider()
            st.subheader("Comparison Insight")

            if result_a["prediction"] != result_b["prediction"]:
                st.write(
                    f"These responses exhibit **different internal deception patterns**: "
                    f"`{result_a['prediction']}` vs `{result_b['prediction']}`."
                )
            else:
                st.write(
                    "Both responses exhibit **similar internal deception patterns**."
                )

        st.caption(
            "This tool surfaces internal patterns, not intent or truthfulness."
        )
