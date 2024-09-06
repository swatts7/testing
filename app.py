import streamlit as st
import pandas as pd
from openai_utils import chat_with_model, calculate_message_cost

st.title("GPT Model Interaction App")

# Use the API key from secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Model selection
model = st.selectbox("Select GPT model:", ["gpt-4o-mini", "gpt-4o"])

# User input
user_message = st.text_area("Enter your message:")

if st.button("Send"):
    if not user_message:
        st.error("Please enter a message.")
    else:
        with st.spinner("Generating response..."):
            response, usage_stats, cost = chat_with_model(api_key, None, user_message, model, "text")
            
            if response:
                st.subheader("Assistant's Response:")
                st.write(response)
                
                st.subheader("Token Usage:")
                usage_df = pd.DataFrame({
                    "Metric": ["Prompt Tokens", "Completion Tokens", "Total Tokens", "Cost"],
                    "Value": [
                        usage_stats.prompt_tokens,
                        usage_stats.completion_tokens,
                        usage_stats.total_tokens,
                        f"${cost:.6f}"
                    ]
                })
                st.table(usage_df)
            else:
                st.error("Failed to get a response. Please check the API key in the secrets file.")

st.sidebar.markdown("""
## How to use this app:
1. Select the GPT model you want to use
2. Type your message in the text area
3. Click 'Send' to get a response
4. View the assistant's response and token usage information
""")
