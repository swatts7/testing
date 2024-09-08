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
                      st.session_state.results.get(selected_operator, ''), 
                      height=200, 
                      key="output_area")

# Generate Summary button
if st.button("Generate Summary"):
    messages = [
        {"role": "system", "content": st.session_state.system_prompt},
        {"role": "user", "content": json.dumps(user_input)}
    ]
    
    response, _, _ = chat_with_model(st.secrets["OPENAI_API_KEY"], messages, None, "gpt-4o", "text")
    
    if response:
        output = response
        st.session_state.results[selected_operator] = output
        st.rerun()
    else:
        st.error("Failed to generate summary. Please try again.")

# Save and Complete button
if st.button("Save and Complete"):
    st.session_state.results[selected_operator] = output
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
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(['Operator', 'Summary'])
        for operator, summary in st.session_state.results.items():
            writer.writerow([operator, summary])
        
        csv_string = csv_buffer.getvalue()
        
        st.sidebar.download_button(
            label="Download CSV",
            data=csv_string,
            file_name=f"optimized_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        st.sidebar.success("CSV file ready for download!")
    else:
        st.sidebar.warning("No results to export yet.")
