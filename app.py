import streamlit as st
import pandas as pd
import json
from openai_utils import chat_with_model

st.title("Master Summary Fine Tuning")

# Initialize session state for system prompt if it doesn't exist
if 'system_prompt' not in st.session_state:
    with open("system_prompt.txt", "r") as file:
        st.session_state.system_prompt = file.read()

# Load the CSV file
@st.cache_data
def load_data():
    return pd.read_csv("master_summary_data.csv")

df = load_data()

# Display editable system prompt in an accordion
with st.expander("Edit System Prompt"):
    edited_system_prompt = st.text_area("System Prompt", st.session_state.system_prompt, height=300, key="system_prompt_input")
    
    # Update session state when the system prompt is edited
    if edited_system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = edited_system_prompt

# Dropdown for operator selection
operator_names = df['operator_name'].tolist()
selected_operator = st.selectbox("Select an operator:", operator_names)

# Get the data for the selected operator
selected_data = df[df['operator_name'] == selected_operator].iloc[0]

# Create the user input JSON
user_input = {
    "casino_name": selected_data['operator_name'],
    "players_summary": selected_data['comment_meta_summary'],
    "experts_summary": selected_data['review_meta_summary']
}

# Display the user input
st.subheader("User Input")
st.text_area("JSON Input", json.dumps(user_input, indent=2), height=200)

# Output text area
st.subheader("Output")
output = st.empty()

# Generate Summary button
if st.button("Generate Summary"):
    # Prepare the messages for the API call
    messages = [
        {"role": "system", "content": st.session_state.system_prompt},
        {"role": "user", "content": json.dumps(user_input)}
    ]
    
    # Call the OpenAI API
    response, usage_stats, _ = chat_with_model(st.secrets["OPENAI_API_KEY"], messages, None, "gpt-4o", "text")
    
    if response:
        output.text_area("Generated Summary", response, height=200)
        
        st.subheader("Token Usage:")
        usage_df = pd.DataFrame({
            "Metric": ["Prompt Tokens", "Completion Tokens", "Total Tokens"],
            "Value": [
                usage_stats.prompt_tokens,
                usage_stats.completion_tokens,
                usage_stats.total_tokens
            ]
        })
        st.table(usage_df)
    else:
        st.error("Failed to generate summary. Please try again.")
