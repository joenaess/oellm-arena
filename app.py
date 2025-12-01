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

# --- EXAMPLE PROMPTS DATABASE ---
EXAMPLE_PROMPTS = {
    "Icelandic": [
        "Einu sinni var lítill strákur sem bjó í ",
        "Helstu einkenni íslenskrar náttúru eru ",
        "Hér er uppskrift að góðum pönnukökum: "
    ],
    "Swedish": [
        "Det var en gång en gammal stuga mitt i ",
        "Det viktigaste för att lyckas med studier är att ",
        "Ingredienser för en klassisk kladdkaka: "
    ],
    "Danish": [
        "Der var engang en konge, som boede i et slot lavet af ",
        "København er kendt for mange ting, blandt andet ",
        "Her er en liste over ting, man skal huske til strandturen: "
    ],
    "Norwegian (Bokmål)": [
        "Langt mot nord, der vinteren varer lenge, bodde det ",
        "Oljefondet har hatt stor betydning for norsk økonomi fordi ",
        "Slik lager du verdens beste vafler: "
    ],
    "Finnish": [
        "Olipa kerran kaukaisessa metsässä pieni ",
        "Suomen koulujärjestelmä on tunnettu siitä, että ",
        "Tässä on resepti perinteiseen karjalanpiirakkaan: "
    ],
    "German": [
        "Es war einmal ein Ritter, der wollte ",
        "Die wichtigste Erfindung des 21. Jahrhunderts ist ",
        "Zutaten für einen perfekten Apfelstrudel: "
    ],
    "Dutch": [
        "Er was eens een kleine kat die hield van ",
        "Amsterdam is een stad vol grachten en ",
        "Het recept voor de beste stroopwafels begint met: "
    ],
    "Spanish": [
        "Había una vez en un pueblo lejano ",
        "La importancia de la dieta mediterránea radica en ",
        "Lista de ingredientes para una paella valenciana: "
    ],
    "Italian": [
        "C'era una volta un falegname che viveva ",
        "Il Rinascimento è stato un periodo cruciale perché ",
        "Per preparare una vera pizza napoletana serve: "
    ],
    "Portuguese": [
        "Era uma vez um navegador que sonhava em ",
        "O fado é uma música tradicional que expressa ",
        "Ingredientes para um bolo de cenoura com chocolate: "
    ],
    "Romanian": [
        "A fost odată ca niciodată un împărat care ",
        "Delta Dunării este un loc unic în Europa datorită ",
        "Rețetă pentru mămăligă cu brânză și smântână: "
    ],
    "Catalan": [
        "Hi havia una vegada un drac que vivia a ",
        "Barcelona és famosa per la seva arquitectura i ",
        "Ingredients per fer pa amb tomàquet: "
    ],
    "Basque": [
        "Bazen behin, mendi altu baten gailurrean, ",
        "Euskararen jatorria ezezaguna da, baina ",
        "Marmitakoa prestatzeko osagaiak hauek dira: "
    ]
}

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
    st.session_state.current_language = "Swedish" # Default
if 'prompt_text' not in st.session_state:
    st.session_state.prompt_text = ""

# --- CALLBACKS (CRITICAL FOR STABILITY) ---
def update_language():
    """Reset everything when language changes"""
    st.session_state.generated = False
    st.session_state.vote_submitted = False
    st.session_state.prompt_text = "" 

def reset_round():
    """Reset everything for a new round"""
    st.session_state.generated = False
    st.session_state.vote_submitted = False
    st.session_state.prompt_text = ""

def set_prompt_callback(text):
    """Set prompt safely"""
    st.session_state.prompt_text = text

# --- SIDEBAR: SETUP ---
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("Choose a language to begin.")

# Select Language with CALLBACK
st.sidebar.selectbox(
    "Select Language", 
    sorted(list(MODELS_DB.keys())),
    key="current_language", 
    on_change=update_language 
)
selected_language = st.session_state.current_language

st.sidebar.divider()

# --- ADVANCED PARAMETERS ---
with st.sidebar.expander("🛠️ Generation Settings", expanded=False):
    st.caption("Adjust these to ensure fair comparisons.")
    min_tokens = st.slider("Min New Tokens", 10, 100, 30)
    max_tokens = st.slider("Max New Tokens", 50, 512, 256)
    rep_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.2, step=0.05)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.1)

# --- MAIN PAGE ---
st.title("⚔️ OELLM Arena")
st.markdown(f"**Current Language:** {selected_language}")

# --- 1. DISCLAIMER ---
with st.expander("ℹ️ **READ THIS FIRST: How to prompt Base Models**", expanded=True):
    st.markdown("""
    **These are Monolingual Base Models, not Chatbots.**
    * 🚫 **Don't ask questions** (e.g., *"What is the capital of Sweden?"*).
    * ✅ **DO write the start of a sentence** (e.g., *"The capital of Sweden is Stockholm, which is famous for..."*).
    * **Think "Auto-Complete":** You start the story, the AI finishes it.
    """)

# --- 2. EXAMPLE PROMPTS ---
st.markdown("### ✍️ Start writing or choose an example")
cols = st.columns(3)
example_list = EXAMPLE_PROMPTS.get(selected_language, ["", "", ""])

# Buttons with CALLBACKS
if cols[0].button("📖 Story Starter", use_container_width=True):
    set_prompt_callback(example_list[0])
if cols[1].button("🧠 Fact Completion", use_container_width=True):
    set_prompt_callback(example_list[1])
if cols[2].button("🍳 Recipe/List", use_container_width=True):
    set_prompt_callback(example_list[2])

# --- 3. INPUT AREA ---
user_prompt = st.text_area("Your Prompt (Start a sentence...):", key="prompt_text", height=100)

if st.button("Generate Response", type="primary"):
    if not user_prompt:
        st.error("Please enter a prompt first.")
    else:
        st.session_state.vote_submitted = False
        st.session_state.last_winner = ""
        
        with st.spinner('Selecting models and generating...'):
            # 1. Randomize
            st.session_state.swap_models = random.choice([True, False])
            
            # 2. Select Models
            multisynt_options = MODELS_DB[selected_language]['multisynt']
            chosen_multisynt = random.choice(multisynt_options)
            chosen_hplt = MODELS_DB[selected_language]['hplt']
            
            # 3. Generate
            try:
                # GPU 0 -> MultiSynt
                pipe_a = get_pipeline(chosen_multisynt, 0)
                res_a = generate_text(pipe_a, user_prompt, 
                                    min_new_tokens=min_tokens,
                                    max_new_tokens=max_tokens,
                                    repetition_penalty=rep_penalty,
                                    temperature=temperature)
                
                # GPU 1 -> HPLT
                pipe_b = get_pipeline(chosen_hplt, 1)
                res_b = generate_text(pipe_b, user_prompt,
                                    min_new_tokens=min_tokens,
                                    max_new_tokens=max_tokens,
                                    repetition_penalty=rep_penalty,
                                    temperature=temperature)
                
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
    
    st.markdown("### 🗳️ Vote for Fluency")
    st.caption("Which model produced the most **grammatically correct** and **natural-sounding** continuation? Ignore factual errors.")

    col1, col2 = st.columns(2)
    
    if st.session_state.swap_models:
        left_text = st.session_state.output_b
        right_text = st.session_state.output_a
    else:
        left_text = st.session_state.output_a
        right_text = st.session_state.output_b

    with col1:
        st.info(left_text)
        if st.button("👈 Better Fluency (Model 1)", key="vote_left", use_container_width=True):
            winner_source = "HPLT" if st.session_state.swap_models else "MultiSynt"
            st.session_state.last_winner = winner_source
            st.session_state.vote_submitted = True
            st.rerun()

    with col2:
        st.info(right_text)
        if st.button("Better Fluency (Model 2) 👉", key="vote_right", use_container_width=True):
            winner_source = "MultiSynt" if st.session_state.swap_models else "HPLT"
            st.session_state.last_winner = winner_source
            st.session_state.vote_submitted = True
            st.rerun()
        
    if st.button("🤝 Tie / Equal Fluency", key="vote_tie", use_container_width=True):
        st.session_state.last_winner = "Tie"
        st.session_state.vote_submitted = True
        st.rerun()

# --- REVEAL & SAVE ---
if st.session_state.vote_submitted:
    vote_val = st.session_state.last_winner
    
    vote_position = "Tie"
    if vote_val != "Tie":
        if vote_val == "MultiSynt":
            vote_position = "Right" if st.session_state.swap_models else "Left"
        else:
            vote_position = "Left" if st.session_state.swap_models else "Right"

    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Language", "Prompt", "Model_A_Name", "Model_B_Name", "Output_A", "Output_B", "Swapped", "Winner_Position", "Winner_Source"])
        
        writer.writerow([
            datetime.now(), 
            st.session_state.current_language, 
            st.session_state.prompt_text, 
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
    
    # CRITICAL FIX: The button now triggers the reset callback instead of running unsafe logic
    st.button("Start New Round", type="primary", on_click=reset_round)

# --- RESULTS ANALYSIS ---
st.divider()
st.subheader("📊 Results & Analytics")

if os.path.exists(RESULTS_FILE):
    df = pd.read_csv(RESULTS_FILE)
    if not df.empty:
        clean_df = df[df['Winner_Source'] != 'Tie']
        
        if not clean_df.empty:
            st.markdown("### 1. Overall Fluency Preference")
            overall_counts = clean_df['Winner_Source'].value_counts()
            col1, col2 = st.columns([2, 1])
            with col1:
                st.bar_chart(overall_counts, color="#FF4B4B")
            with col2:
                st.write(overall_counts)
                st.caption(f"Total valid votes: {len(clean_df)}")

            st.divider()
            st.markdown("### 2. Preference by Language")
            lang_stats = pd.crosstab(clean_df['Language'], clean_df['Winner_Source'])
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.bar_chart(lang_stats)
            with col_b:
                st.dataframe(lang_stats)

            st.divider()
            st.markdown("### 3. MultiSynt Type Preference")
            def extract_type(name):
                if pd.isna(name): return "Unknown"
                if "opus" in name.lower(): return "Opus"
                if "tower" in name.lower(): return "Tower"
                return "Other"
            
            ms_wins = clean_df[clean_df['Winner_Source'] == 'MultiSynt'].copy()
            if not ms_wins.empty:
                ms_wins['Architecture'] = ms_wins['Model_A_Name'].apply(extract_type)
                type_counts = ms_wins['Architecture'].value_counts()
                col_x, col_y = st.columns([2, 1])
                with col_x:
                    st.bar_chart(type_counts, color="#1E90FF")
                with col_y:
                    st.write(type_counts)
            else:
                st.info("No MultiSynt wins recorded yet.")
        else:
            st.write("No clear votes yet (only ties).")
    else:
        st.write("Dataset is empty.")
else:
    st.write("No votes recorded yet.")