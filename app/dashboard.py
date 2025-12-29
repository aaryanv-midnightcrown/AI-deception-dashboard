import streamlit as st
from core.analyze import analyze_response

st.set_page_config(page_title="AI Deception Dashboard", layout="centered")

st.title("AI Deception Dashboard")
st.write(
    "This tool explores internal signals associated with different deceptive behaviors in AI responses."
)

prompt = st.text_area("User Prompt")
response = st.text_area("AI Response")

if st.button("Analyze"):
    if not prompt or not response:
        st.warning("Please enter both a prompt and a response.")
    else:
        result = analyze_response(prompt, response)

        st.subheader("Result")
        st.write(f"**Predicted behavior:** {result['prediction']}")
        st.write(f"**Confidence:** {result['confidence']}")

        st.subheader("Probabilities")
        st.json(result["probabilities"])

        st.subheader("Mechanistic Insight")
        st.write(
            f"Top contributing internal feature: `{result['top_feature_index']}`"
        )
        st.write(
            f"Estimated impact if removed: `{result['ablation_effect']}`"
        )

        st.caption(
            "Note: This is a research demo. Signals indicate patterns, not intent."
        )
