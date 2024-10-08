import streamlit as st
import pandas as pd
import json
from openai_utils import chat_with_model
import csv
from datetime import datetime
import io
import os

st.title("Dataset Fine Tuning")

# Initialize session state
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = {}

if 'completed_operators' not in st.session_state:
    st.session_state.completed_operators = {}

if 'results' not in st.session_state:
    st.session_state.results = {}

if 'current_operator_index' not in st.session_state:
    st.session_state.current_operator_index = {}

# Dataset selection
dataset_type = st.selectbox("Select Dataset:", ["Master Summary", "Comment Summary", "Reviews Summary"])

# Load data based on dataset type
@st.cache_data
def load_data(dataset_type):
    if dataset_type == "Master Summary":
        return pd.read_csv("master_summary_data.csv")
    elif dataset_type == "Comment Summary":
        with open("comments_data.json", "r") as f:
            return json.load(f)
    elif dataset_type == "Reviews Summary":
        with open("reviews.json", "r") as f:
            return json.load(f)

data = load_data(dataset_type)

# Load system prompt
if dataset_type not in st.session_state.system_prompt:
    if dataset_type == "Master Summary":
        file_path = "master_system_prompt.txt"
    elif dataset_type == "Comment Summary":
        file_path = "comments_system_prompt.txt"
    elif dataset_type == "Reviews Summary":
        file_path = "reviews_system_prompt.txt"
    
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            st.session_state.system_prompt[dataset_type] = file.read()
    else:
        st.error(f"System prompt file not found: {file_path}")
        st.session_state.system_prompt[dataset_type] = ""

# Initialize dataset-specific session state
if dataset_type not in st.session_state.completed_operators:
    st.session_state.completed_operators[dataset_type] = set()

if dataset_type not in st.session_state.results:
    st.session_state.results[dataset_type] = {}

if dataset_type not in st.session_state.current_operator_index:
    st.session_state.current_operator_index[dataset_type] = 0

# Display editable system prompt in an accordion
with st.expander("Edit System Prompt"):
    edited_system_prompt = st.text_area("System Prompt", st.session_state.system_prompt[dataset_type], height=300, key=f"system_prompt_input_{dataset_type}")
    if edited_system_prompt != st.session_state.system_prompt[dataset_type]:
        st.session_state.system_prompt[dataset_type] = edited_system_prompt

# Prepare operator names
if dataset_type == "Master Summary":
    operator_names = data['operator_name'].tolist()
elif dataset_type == "Comment Summary":
    operator_names = [item['casino_name'] for item in data.values()]
elif dataset_type == "Reviews Summary":
    operator_names = [f"Operator {item['operator_id']}" for item in data]

# Dropdown for operator selection
selected_operator = st.selectbox("Select an operator:", operator_names, index=st.session_state.current_operator_index[dataset_type])

# Next button
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Next"):
        st.session_state.current_operator_index[dataset_type] = (st.session_state.current_operator_index[dataset_type] + 1) % len(operator_names)
        st.rerun()

# Get the data for the selected operator
if dataset_type == "Master Summary":
    selected_data = data[data['operator_name'] == selected_operator].iloc[0]
    operator_id = selected_data['operator_id']
    user_input = json.dumps({
        "casino_name": selected_data['operator_name'],
        "players_summary": selected_data['comment_meta_summary'],
        "experts_summary": selected_data['review_meta_summary']
    })
elif dataset_type == "Comment Summary":
    selected_data = next(item for item in data.values() if item['casino_name'] == selected_operator)
    operator_id = next(key for key in data.keys() if data[key]['casino_name'] == selected_operator)
    user_input = json.dumps(selected_data)
elif dataset_type == "Reviews Summary":
    selected_data = next(item for item in data if f"Operator {item['operator_id']}" == selected_operator)
    operator_id = selected_data['operator_id']
    user_input = json.dumps(selected_data['unique_reviews'])

# Display the user input
st.subheader("User Input")
st.text_area("JSON Input", user_input, height=200)

# Output text area
st.subheader("Output")
if selected_operator not in st.session_state.results[dataset_type]:
    st.session_state.results[dataset_type][selected_operator] = ''

output = st.text_area("Generated Summary", 
                      st.session_state.results[dataset_type][selected_operator], 
                      height=200, 
                      key=f"output_area_{dataset_type}")

# Generate Summary button
if st.button("Generate Summary"):
    messages = [
        {"role": "system", "content": st.session_state.system_prompt[dataset_type]},
        {"role": "user", "content": user_input}
    ]
    
    response, _, _ = chat_with_model(st.secrets["OPENAI_API_KEY"], messages, None, "gpt-4o-mini", "text")
    
    if response:
        st.session_state.results[dataset_type][selected_operator] = response
        st.rerun()
    else:
        st.error("Failed to generate summary. Please try again.")

# Save and Complete button
if st.button("Save and Complete"):
    st.session_state.results[dataset_type][selected_operator] = output
    st.session_state.completed_operators[dataset_type].add(selected_operator)
    st.success(f"Saved and completed summary for {selected_operator}")

# Progress tracking
total_operators = len(operator_names)
completed_operators = len(st.session_state.completed_operators[dataset_type])
st.sidebar.progress(completed_operators / total_operators)
st.sidebar.write(f"Completed: {completed_operators}/{total_operators}")

# Export results button
if st.sidebar.button("Export Results"):
    if st.session_state.results[dataset_type]:
        # Prepare CSV
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(['operator_id', 'operator_name', 'system_prompt', 'user_input', 'output'])
        
        # Prepare JSONL
        jsonl_buffer = io.StringIO()
        
        for operator, summary in st.session_state.results[dataset_type].items():
            # Get operator data
            if dataset_type == "Master Summary":
                operator_data = data[data['operator_name'] == operator].iloc[0]
                operator_id = operator_data['operator_id']
                user_input = json.dumps({
                    "casino_name": operator_data['operator_name'],
                    "players_summary": operator_data['comment_meta_summary'],
                    "experts_summary": operator_data['review_meta_summary']
                })
            elif dataset_type == "Comment Summary":
                operator_data = next(item for item in data.values() if item['casino_name'] == operator)
                operator_id = next(key for key in data.keys() if data[key]['casino_name'] == operator)
                user_input = json.dumps(operator_data)
            elif dataset_type == "Reviews Summary":
                operator_data = next(item for item in data if f"Operator {item['operator_id']}" == operator)
                operator_id = operator_data['operator_id']
                user_input = json.dumps(operator_data['unique_reviews'])
            
            # Write to CSV
            writer.writerow([operator_id, operator, st.session_state.system_prompt[dataset_type], user_input, summary])
            
            # Write to JSONL
            jsonl_entry = {
                "messages": [
                    {"role": "system", "content": st.session_state.system_prompt[dataset_type]},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": summary}
                ]
            }
            jsonl_buffer.write(json.dumps(jsonl_entry) + '\n')
        
        csv_string = csv_buffer.getvalue()
        jsonl_string = jsonl_buffer.getvalue()
        
        # Create download buttons
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.download_button(
                label="Download CSV",
                data=csv_string,
                file_name=f"{dataset_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                label="Download JSONL",
                data=jsonl_string,
                file_name=f"{dataset_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                mime="application/jsonl"
            )
        
        st.sidebar.success("CSV and JSONL files ready for download!")
    else:
        st.sidebar.warning("No results to export yet.")
