import streamlit as st
import pandas as pd
import json
from openai_utils import chat_with_model
import csv
from datetime import datetime

st.title("Master Summary Fine Tuning")

# Initialize session state
if 'system_prompt' not in st.session_state:
    with open("system_prompt.txt", "r") as file:
        st.session_state.system_prompt = file.read()

if 'completed_operators' not in st.session_state:
    st.session_state.completed_operators = set()

if 'results' not in st.session_state:
    st.session_state.results = {}

# Load the CSV file
@st.cache_data
def load_data():
    return pd.read_csv("master_summary_data.csv")

df = load_data()

# Display editable system prompt in an accordion
with st.expander("Edit System Prompt"):
    edited_system_prompt = st.text_area("System Prompt", st.session_state.system_prompt, height=300, key="system_prompt_input")
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
output = st.text_area("Generated Summary", 
                      st.session_state.results.get(selected_operator, {}).get('summary', ''), 
                      height=200, 
                      key="output_area")

# Generate Summary button
if st.button("Generate Summary"):
    messages = [
        {"role": "system", "content": st.session_state.system_prompt},
        {"role": "user", "content": json.dumps(user_input)}
    ]
    
    response, usage_stats, _ = chat_with_model(st.secrets["OPENAI_API_KEY"], messages, None, "gpt-4o", "text")
    
    if response:
        output = response
        st.session_state.results[selected_operator] = {
            'summary': output,
            'usage_stats': usage_stats._asdict()
        }
        st.experimental_rerun()
    else:
        st.error("Failed to generate summary. Please try again.")

# Save and Complete button
if st.button("Save and Complete"):
    st.session_state.results[selected_operator] = {
        'summary': output,
        'usage_stats': st.session_state.results.get(selected_operator, {}).get('usage_stats', {})
    }
    st.session_state.completed_operators.add(selected_operator)
    st.success(f"Saved and completed summary for {selected_operator}")

# Progress tracking
total_operators = len(operator_names)
completed_operators = len(st.session_state.completed_operators)
st.sidebar.progress(completed_operators / total_operators)
st.sidebar.write(f"Completed: {completed_operators}/{total_operators}")

# Export results button
if st.sidebar.button("Export Results"):
    if st.session_state.results:
        csv_filename = f"optimized_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Operator', 'Summary', 'Prompt Tokens', 'Completion Tokens', 'Total Tokens'])
            for operator, data in st.session_state.results.items():
                writer.writerow([
                    operator,
                    data['summary'],
                    data['usage_stats'].get('prompt_tokens', ''),
                    data['usage_stats'].get('completion_tokens', ''),
                    data['usage_stats'].get('total_tokens', '')
                ])
        st.sidebar.success(f"Results exported to {csv_filename}")
    else:
        st.sidebar.warning("No results to export yet.")

# Display token usage for current operator
if selected_operator in st.session_state.results:
    usage_stats = st.session_state.results[selected_operator]['usage_stats']
    st.subheader("Token Usage:")
    usage_df = pd.DataFrame({
        "Metric": ["Prompt Tokens", "Completion Tokens", "Total Tokens"],
        "Value": [
            usage_stats.get('prompt_tokens', ''),
            usage_stats.get('completion_tokens', ''),
            usage_stats.get('total_tokens', '')
        ]
    })
    st.table(usage_df)
