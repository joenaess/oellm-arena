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
if 'current_language' not in st.session_state:
    st.session_state.current_language = ""

# --- SIDEBAR: SETUP ---
st.sidebar.title("⚙️ Language Setup")
st.sidebar.markdown("Choose a language to begin.")

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
        st.session_state.vote_submitted = False
        st.session_state.last_winner = ""
        st.session_state.current_language = selected_language
        
        with st.spinner('Selecting models and generating...'):
            # 1. Randomize display order
            st.session_state.swap_models = random.choice([True, False])
            
            # 2. Select Models
            multisynt_options = MODELS_DB[selected_language]['multisynt']
            chosen_multisynt = random.choice(multisynt_options)
            chosen_hplt = MODELS_DB[selected_language]['hplt']
            
            # 3. Generate
            try:
                # Load A (MultiSynt) on GPU 0
                pipe_a = get_pipeline(chosen_multisynt, 0)
                res_a = generate_text(pipe_a, user_prompt)
                
                # Load B (HPLT) on GPU 1
                pipe_b = get_pipeline(chosen_hplt, 1)
                res_b = generate_text(pipe_b, user_prompt)
                
                st.session_state.model_a_name = chosen_multisynt
                st.session_state.model_b_name = chosen_hplt
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
if st.session_state.generated and not st.session_state.vote_submitted:
    st.divider()
    col1, col2 = st.columns(2)
    
    if st.session_state.swap_models:
        left_text = st.session_state.output_b
        right_text = st.session_state.output_a
    else:
        left_text = st.session_state.output_a
        right_text = st.session_state.output_b

    with col1:
        st.subheader("Model 1")
        st.info(left_text)
        if st.button("👈 Vote for Model 1", key="vote_left"):
            winner_source = "HPLT" if st.session_state.swap_models else "MultiSynt"
            st.session_state.last_winner = winner_source
            st.session_state.vote_submitted = True
            st.rerun()

    with col2:
        st.subheader("Model 2")
        st.info(right_text)
        if st.button("Vote for Model 2 👉", key="vote_right"):
            winner_source = "MultiSynt" if st.session_state.swap_models else "HPLT"
            st.session_state.last_winner = winner_source
            st.session_state.vote_submitted = True
            st.rerun()
        
    if st.button("🤝 Tie / Both Good / Both Bad", key="vote_tie"):
        st.session_state.last_winner = "Tie"
        st.session_state.vote_submitted = True
        st.rerun()

# --- REVEAL & SAVE ---
if st.session_state.vote_submitted:
    vote_val = st.session_state.last_winner
    
    # Calculate position for log
    vote_position = "Tie"
    if vote_val != "Tie":
        if vote_val == "MultiSynt":
            vote_position = "Right" if st.session_state.swap_models else "Left"
        else:
            vote_position = "Left" if st.session_state.swap_models else "Right"

    # Save Data
    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Added 'Language' column to header
        if not file_exists:
            writer.writerow(["Timestamp", "Language", "Prompt", "Model_A_Name", "Model_B_Name", "Output_A", "Output_B", "Swapped", "Winner_Position", "Winner_Source"])
        
        writer.writerow([
            datetime.now(), 
            st.session_state.current_language, # Save the language!
            user_prompt, 
            st.session_state.model_a_name, 
            st.session_state.model_b_name, 
            st.session_state.output_a, 
            st.session_state.output_b, 
            st.session_state.swap_models, 
            vote_position, 
            vote_val
        ])

    st.divider()
    if vote_val == "MultiSynt":
        st.success("🎉 You preferred MultiSynt!")
    elif vote_val == "HPLT":
        st.error("🎉 You preferred HPLT!")
    else:
        st.warning("🤝 It was a tie.")
        
    st.markdown("### 🕵️ Identity Reveal")
    col1, col2 = st.columns(2)
    
    def clean_name(name):
        return name.split("/")[-1]

    with col1:
        st.markdown(f"**MultiSynt Model (A)**")
        st.code(clean_name(st.session_state.model_a_name))
            
    with col2:
        st.markdown(f"**HPLT Model (B)**")
        st.code(clean_name(st.session_state.model_b_name))
            
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
        # Filter out ties for win-rate calculations
        clean_df = df[df['Winner_Source'] != 'Tie']
        
        if not clean_df.empty:
            
            # --- 1. OVERALL ---
            st.markdown("### 1. Overall Win Rate")
            overall_counts = clean_df['Winner_Source'].value_counts()
            col1, col2 = st.columns([2, 1])
            with col1:
                st.bar_chart(overall_counts, color="#FF4B4B")
            with col2:
                st.write(overall_counts)
                st.caption(f"Total valid votes: {len(clean_df)}")

            st.divider()

            # --- 2. PER LANGUAGE ---
            st.markdown("### 2. Win Rate by Language")
            
            # Create a pivot table: Language as rows, Winner_Source as columns
            lang_stats = pd.crosstab(clean_df['Language'], clean_df['Winner_Source'])
            
            # Calculate percentages for tooltips/display
            lang_pct = lang_stats.div(lang_stats.sum(1), axis=0) * 100
            
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.bar_chart(lang_stats) # Stacked bar chart by default
            with col_b:
                st.dataframe(lang_stats) # Show raw counts
                st.caption("Vote counts per language")

            st.divider()

            # --- 3. MULTISYNT TYPE PREFERENCE ---
            st.markdown("### 3. MultiSynt Type Preference (Opus vs Tower)")
            
            def extract_type(name):
                if pd.isna(name): return "Unknown"
                if "opus" in name.lower(): return "Opus"
                if "tower" in name.lower(): return "Tower"
                return "Other"
            
            # Filter only MultiSynt wins to see which type wins more often
            ms_wins = clean_df[clean_df['Winner_Source'] == 'MultiSynt'].copy()
            
            if not ms_wins.empty:
                ms_wins['Architecture'] = ms_wins['Model_A_Name'].apply(extract_type)
                type_counts = ms_wins['Architecture'].value_counts()
                
                col_x, col_y = st.columns([2, 1])
                with col_x:
                    st.bar_chart(type_counts, color="#1E90FF")
                with col_y:
                    st.write(type_counts)
                    st.caption("Total MultiSynt Wins by Type")
            else:
                st.info("No MultiSynt wins recorded yet.")

        else:
            st.write("No clear votes yet (only ties).")
    else:
        st.write("Dataset is empty.")
else:
    st.write("No votes recorded yet.")