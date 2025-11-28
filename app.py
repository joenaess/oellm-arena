import streamlit as st
import pandas as pd
import random
import os
import csv
from datetime import datetime
from backend import MULTI_SYNT_MODELS, LANGUAGE_TO_HPLT, get_pipeline, generate_text

# --- CONFIGURATION ---
RESULTS_FILE = "arena_results.csv"

st.set_page_config(layout="wide", page_title="OELLM Arena")

# --- INITIALIZE SESSION STATE ---
if 'generated' not in st.session_state:
    st.session_state.generated = False
if 'output_a' not in st.session_state:
    st.session_state.output_a = ""
if 'output_b' not in st.session_state:
    st.session_state.output_b = ""
if 'model_a_name' not in st.session_state:
    st.session_state.model_a_name = ""
if 'model_b_name' not in st.session_state:
    st.session_state.model_b_name = ""
if 'swap_models' not in st.session_state:
    st.session_state.swap_models = False 

# --- HELPER: EXTRACT LANGUAGE ---
def get_hplt_options(multisynt_name):
    """
    Parses 'MultiSynt/nemotron-cc-swedish-tower9b' to find 'swedish',
    then looks up the corresponding HPLT model.
    """
    try:
        # Expected format: MultiSynt/nemotron-cc-{LANGUAGE}-{TYPE}
        parts = multisynt_name.split("nemotron-cc-")
        if len(parts) > 1:
            rest = parts[1]
            language = rest.split("-")[0]
            
            if language in LANGUAGE_TO_HPLT:
                return [LANGUAGE_TO_HPLT[language]]
            
        return [] 
    except Exception:
        return []

# --- SIDEBAR: SETUP ---
st.sidebar.title("⚙️ Model Setup")
st.sidebar.markdown("Select models to compare.")

selected_multisynt = st.sidebar.selectbox(
    "1. Select MultiSynt Model (Model A)", 
    MULTI_SYNT_MODELS
)

hplt_options = get_hplt_options(selected_multisynt)

if hplt_options:
    selected_hplt = st.sidebar.selectbox(
        "2. Select Matching HPLT Model (Model B)", 
        hplt_options
    )
    st.sidebar.success(f"Matched language: {hplt_options[0].split('_')[1].upper()}")
else:
    selected_hplt = st.sidebar.selectbox(
        "2. Select HPLT Model (Model B)", 
        ["No matching HPLT model found"], 
        disabled=True
    )
    st.sidebar.warning("Could not auto-match an HPLT model for this language.")

# --- MAIN PAGE: GENERATION ---
st.title("⚔️ OELLM Arena")
st.markdown("Enter a prompt. The system will generate two responses. Vote for the best one.")

user_prompt = st.text_area("Enter your prompt:", height=100)

can_generate = selected_hplt and "No matching" not in selected_hplt

if st.button("Generate A & B", disabled=not can_generate):
    if not user_prompt:
        st.error("Please enter a prompt first.")
    else:
        with st.spinner('Loading models and generating text...'):
            st.session_state.swap_models = random.choice([True, False])
            
            try:
                # Load A (MultiSynt) on GPU 0
                pipe_a = get_pipeline(selected_multisynt, 0)
                res_a = generate_text(pipe_a, user_prompt)
                
                # Load B (HPLT) on GPU 1
                pipe_b = get_pipeline(selected_hplt, 1)
                res_b = generate_text(pipe_b, user_prompt)
                
                st.session_state.model_a_name = selected_multisynt
                st.session_state.model_b_name = selected_hplt
                st.session_state.output_a = res_a
                st.session_state.output_b = res_b
                st.session_state.generated = True
                
                del pipe_a
                del pipe_b
                import torch
                torch.cuda.empty_cache()

            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- DISPLAY & VOTING ---
if st.session_state.generated:
    st.divider()
    col1, col2 = st.columns(2)
    
    if st.session_state.swap_models:
        left_text, left_source = st.session_state.output_b, "HPLT"
        right_text, right_source = st.session_state.output_a, "MultiSynt"
    else:
        left_text, left_source = st.session_state.output_a, "MultiSynt"
        right_text, right_source = st.session_state.output_b, "HPLT"

    with col1:
        st.subheader("Model 1")
        st.info(left_text)
        btn_left = st.button("👈 Vote for Model 1", key="vote_left")

    with col2:
        st.subheader("Model 2")
        st.info(right_text)
        btn_right = st.button("Vote for Model 2 👉", key="vote_right")
        
    btn_tie = st.button("🤝 Tie / Both Good / Both Bad", key="vote_tie")

    vote_choice = None
    winner_source = None
    
    if btn_left:
        vote_choice, winner_source = "Left", left_source
    elif btn_right:
        vote_choice, winner_source = "Right", right_source
    elif btn_tie:
        vote_choice, winner_source = "Tie", "Tie"

    if vote_choice:
        file_exists = os.path.isfile(RESULTS_FILE)
        with open(RESULTS_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Prompt", "Model_A_Name", "Model_B_Name", "Output_A", "Output_B", "Swapped", "Winner_Position", "Winner_Source"])
            
            writer.writerow([
                datetime.now(), user_prompt, st.session_state.model_a_name, st.session_state.model_b_name, 
                st.session_state.output_a, st.session_state.output_b, st.session_state.swap_models, vote_choice, winner_source
            ])
        
        st.success(f"Vote saved! Preferred: {winner_source}")
        st.session_state.generated = False
        st.rerun()

# --- RESULTS ANALYSIS ---
st.divider()
st.subheader("📊 Results & Analytics")

if os.path.exists(RESULTS_FILE):
    df = pd.read_csv(RESULTS_FILE)
    
    if not df.empty:
        # 1. Main A vs B Chart
        clean_df = df[df['Winner_Source'] != 'Tie']
        
        if not clean_df.empty:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### Overall Preference (MultiSynt vs HPLT)")
                stats = clean_df['Winner_Source'].value_counts()
                st.bar_chart(stats, color="#FF4B4B") # Red for HPLT/MultiSynt Comparison
                
                total = len(clean_df)
                multi_wins = len(clean_df[clean_df['Winner_Source'] == 'MultiSynt'])
                hplt_wins = len(clean_df[clean_df['Winner_Source'] == 'HPLT'])
                st.write(f"**MultiSynt:** {multi_wins} ({multi_wins/total:.1%})")
                st.write(f"**HPLT:** {hplt_wins} ({hplt_wins/total:.1%})")

            # 2. Sub-Category Analysis (Opus vs Tower9b vs Tower72b)
            with col_b:
                st.markdown("#### MultiSynt Performance by Type")
                
                # Extract the type (last part of the name)
                # e.g., "nemotron-cc-swedish-tower9b" -> "tower9b"
                def extract_type(name):
                    if pd.isna(name): return "Unknown"
                    return name.split('-')[-1] # Gets 'opus', 'tower9b', etc.

                # Filter for rows where MultiSynt won OR lost (to see win rate per type)
                # But user asked for "preference between Opus, Tower9b...", so let's look at MultiSynt WINS
                
                multi_wins_df = clean_df[clean_df['Winner_Source'] == 'MultiSynt'].copy()
                
                if not multi_wins_df.empty:
                    multi_wins_df['Model_Type'] = multi_wins_df['Model_A_Name'].apply(extract_type)
                    type_stats = multi_wins_df['Model_Type'].value_counts()
                    
                    st.bar_chart(type_stats, color="#1E90FF") # Blue for Types
                    st.caption("Count of wins per MultiSynt architecture type.")
                else:
                    st.info("No MultiSynt wins recorded yet to analyze types.")

            with st.expander("View Raw Data"):
                st.dataframe(df)
        else:
            st.write("No clear votes yet (only ties).")
    else:
        st.write("Dataset is empty.")
else:
    st.write("No votes recorded yet.")