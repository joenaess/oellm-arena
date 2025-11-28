import streamlit as st
import pandas as pd
import random
import os
import csv
from datetime import datetime
from backend import MODELS_DB, get_pipeline, generate_text

# --- CONFIGURATION ---
RESULTS_FILE = "arena_results.csv"

st.set_page_config(layout="wide", page_title="OELLM Arena")

# --- INITIALIZE SESSION STATE ---
if 'generated' not in st.session_state:
    st.session_state.generated = False
if 'vote_submitted' not in st.session_state:
    st.session_state.vote_submitted = False
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
if 'last_winner' not in st.session_state:
    st.session_state.last_winner = ""

# --- SIDEBAR: SETUP ---
st.sidebar.title("⚙️ Language Setup")
st.sidebar.markdown("Choose a language to begin.")

# 1. Select Language
selected_language = st.sidebar.selectbox(
    "Select Language", 
    sorted(list(MODELS_DB.keys()))
)

# --- MAIN PAGE: GENERATION ---
st.title("⚔️ OELLM Arena")
st.markdown(f"**Language:** {selected_language}")
st.markdown("Enter a prompt. The system will randomly select two AI models. Vote for the best one!")

user_prompt = st.text_area("Enter your prompt:", height=100)

if st.button("Generate Response"):
    if not user_prompt:
        st.error("Please enter a prompt first.")
    else:
        # Reset previous vote state
        st.session_state.vote_submitted = False
        st.session_state.last_winner = ""
        
        with st.spinner('Selecting models and generating...'):
            # 1. Randomize display order (Blind Test)
            st.session_state.swap_models = random.choice([True, False])
            
            # 2. Select Models
            # Randomly pick one MultiSynt model for this language
            multisynt_options = MODELS_DB[selected_language]['multisynt']
            chosen_multisynt = random.choice(multisynt_options)
            
            # Pick the corresponding HPLT model
            chosen_hplt = MODELS_DB[selected_language]['hplt']
            
            # 3. Generate
            try:
                # Load A (MultiSynt) on GPU 0
                pipe_a = get_pipeline(chosen_multisynt, 0)
                res_a = generate_text(pipe_a, user_prompt)
                
                # Load B (HPLT) on GPU 1
                pipe_b = get_pipeline(chosen_hplt, 1)
                res_b = generate_text(pipe_b, user_prompt)
                
                # Store in session (but keep secret from UI for now)
                st.session_state.model_a_name = chosen_multisynt
                st.session_state.model_b_name = chosen_hplt
                st.session_state.output_a = res_a
                st.session_state.output_b = res_b
                st.session_state.generated = True
                
                # Cleanup
                del pipe_a
                del pipe_b
                import torch
                torch.cuda.empty_cache()

            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- DISPLAY & VOTING (BLIND) ---
if st.session_state.generated and not st.session_state.vote_submitted:
    st.divider()
    col1, col2 = st.columns(2)
    
    # Assign text to Left/Right based on swap_models
    if st.session_state.swap_models:
        # Left = HPLT, Right = MultiSynt
        left_text = st.session_state.output_b
        right_text = st.session_state.output_a
    else:
        # Left = MultiSynt, Right = HPLT
        left_text = st.session_state.output_a
        right_text = st.session_state.output_b

    with col1:
        st.subheader("Model 1") # Blind Label
        st.info(left_text)
        if st.button("👈 Vote for Model 1", key="vote_left"):
            vote_choice = "Left"
            if st.session_state.swap_models:
                winner_source = "HPLT"
            else:
                winner_source = "MultiSynt"
            st.session_state.last_winner = winner_source
            st.session_state.vote_submitted = True
            st.rerun()

    with col2:
        st.subheader("Model 2") # Blind Label
        st.info(right_text)
        if st.button("Vote for Model 2 👉", key="vote_right"):
            vote_choice = "Right"
            if st.session_state.swap_models:
                winner_source = "MultiSynt"
            else:
                winner_source = "HPLT"
            st.session_state.last_winner = winner_source
            st.session_state.vote_submitted = True
            st.rerun()
        
    if st.button("🤝 Tie / Both Good / Both Bad", key="vote_tie"):
        vote_choice = "Tie"
        st.session_state.last_winner = "Tie"
        st.session_state.vote_submitted = True
        st.rerun()

# --- REVEAL & SAVE (AFTER VOTE) ---
if st.session_state.vote_submitted:
    # 1. Save Data
    # We only save once per vote submission
    # (In a real production app, we'd handle this to prevent re-saving on refresh, 
    # but st.rerun() above helps manage flow)
    
    vote_val = st.session_state.last_winner
    
    # Calculate who was left/right for the log
    vote_position = "Tie"
    if vote_val != "Tie":
        # If user voted MultiSynt
        if vote_val == "MultiSynt":
            vote_position = "Right" if st.session_state.swap_models else "Left"
        else:
            vote_position = "Left" if st.session_state.swap_models else "Right"

    # Append to CSV
    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Prompt", "Model_A_Name", "Model_B_Name", "Output_A", "Output_B", "Swapped", "Winner_Position", "Winner_Source"])
        
        writer.writerow([
            datetime.now(), user_prompt, st.session_state.model_a_name, st.session_state.model_b_name, 
            st.session_state.output_a, st.session_state.output_b, st.session_state.swap_models, vote_position, vote_val
        ])

    # 2. Show Reveal UI
    st.divider()
    if vote_val == "MultiSynt":
        st.success("🎉 You preferred MultiSynt!")
    elif vote_val == "HPLT":
        st.error("🎉 You preferred HPLT!")
    else:
        st.warning("🤝 It was a tie.")
        
    st.markdown("### 🕵️ Identity Reveal")
    col1, col2 = st.columns(2)
    
    # Helper to strip "MultiSynt/nemotron-cc-" prefix for cleaner display
    def clean_name(name):
        return name.split("/")[-1]

    with col1:
        st.markdown(f"**MultiSynt Model (A)**")
        st.code(clean_name(st.session_state.model_a_name))
        with st.expander("Show Text"):
            st.write(st.session_state.output_a)
            
    with col2:
        st.markdown(f"**HPLT Model (B)**")
        st.code(clean_name(st.session_state.model_b_name))
        with st.expander("Show Text"):
            st.write(st.session_state.output_b)
            
    if st.button("Start New Round"):
        st.session_state.generated = False
        st.session_state.vote_submitted = False
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
                st.markdown("#### Overall Preference")
                stats = clean_df['Winner_Source'].value_counts()
                st.bar_chart(stats, color="#FF4B4B") 
            
            with col_b:
                st.markdown("#### MultiSynt Type Analysis")
                def extract_type(name):
                    if pd.isna(name): return "Unknown"
                    return name.split('-')[-1] # Gets 'opus', 'tower9b', etc.
                
                multi_wins_df = clean_df[clean_df['Winner_Source'] == 'MultiSynt'].copy()
                if not multi_wins_df.empty:
                    multi_wins_df['Model_Type'] = multi_wins_df['Model_A_Name'].apply(extract_type)
                    type_stats = multi_wins_df['Model_Type'].value_counts()
                    st.bar_chart(type_stats, color="#1E90FF")
                else:
                    st.info("No MultiSynt wins yet.")