# -*- coding: utf-8 -*-
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
        "Einu sinni var l\u00edtill str\u00e1kur sem bj\u00f3 \u00ed ",
        "Helstu einkenni \u00edslenskrar n\u00e1tt\u00faru eru ",
        "H\u00e9r er uppskrift a\u00f0 g\u00f3\u00f0um p\u00f6nnuk\u00f6kum: "
    ],
    "Swedish": [
        "Det var en g\u00e5ng en gammal stuga mitt i ",
        "Det viktigaste f\u00f6r att lyckas med studier \u00e4r att ",
        "Ingredienser f\u00f6r en klassisk kladdkaka: "
    ],
    "Danish": [
        "Der var engang en konge, som boede i et slot lavet af ",
        "K\u00f8benhavn er kendt for mange ting, blandt andet ",
        "Her er en liste over ting, man skal huske til strandturen: "
    ],
    "Norwegian": [
        "Langt mot nord, der vinteren varer lenge, bodde det ",
        "Oljefondet har hatt stor betydning for norsk \u00f8konomi fordi ", 
        "Slik lager du verdens beste vafler: "
    ],
    "Finnish": [
        "Olipa kerran kaukaisessa mets\u00e4ss\u00e4 pieni ",
        "Suomen kouluj\u00e4rjestelm\u00e4 on tunnettu siit\u00e4, ett\u00e4 ", 
        "T\u00e4ss\u00e4 on resepti perinteiseen karjalanpiirakkaan: "
    ],
    "German": [
        "Es war einmal ein Ritter, der wollte ",
        "Die wichtigste Erfindung des 21. Jahrhunderts ist ",
        "Zutaten f\u00fcr einen perfekten Apfelstrudel: "
    ],
    "Dutch": [
        "Er was eens een kleine kat die hield van ",
        "Amsterdam is een stad vol grachten en ",
        "Het recept voor de beste stroopwafels begint met: "
    ],
    "Spanish": [
        "Hab\u00eda una vez en un pueblo lejano ",
        "La importancia de la dieta mediterr\u00e1nea radica en ",
        "Lista de ingredientes para una paella valenciana: "
    ],
    "Italian": [
        "C'era una volta un falegname che viveva ",
        "Il Rinascimento \u00e8 stato un periodo cruciale perch\u00e9 ",
        "Per preparare una vera pizza napoletana serve: "
    ],
    "Portuguese": [
        "Era uma vez um navegador que sonhava em ",
        "O fado \u00e9 uma m\u00fasica tradicional que expressa ",
        "Ingredientes para um bolo de cenoura com chocolate: "
    ],
    "Romanian": [
        "A fost odat\u0103 ca niciodat\u0103 un \u00eemp\u0103rat care ",
        "Delta Dun\u0103rii este un loc unic \u00een Europa datorit\u0103 ",
        "Re\u021bet\u0103 pentru m\u0103m\u0103lig\u0103 cu br\u00e2nz\u0103 \u0219i sm\u00e2nt\u00e2n\u0103: "
    ],
    "Catalan": [
        "Hi havia una vegada un drac que vivia a ",
        "Barcelona \u00e9s famosa per la seva arquitectura i ",
        "Ingredients per fer pa amb tom\u00e0quet: "
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

# Session Stats Tracking (For anonymized feedback)
if 'vote_count' not in st.session_state:
    st.session_state.vote_count = 0
if 'session_wins' not in st.session_state:
    st.session_state.session_wins = {"MultiSynt": 0, "HPLT": 0, "Tie": 0}
if 'session_history' not in st.session_state:
    st.session_state.session_history = []

# --- CALLBACKS ---
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

def register_vote(winner_source):
    """Update stats and set state"""
    st.session_state.last_winner = winner_source
    st.session_state.vote_submitted = True
    
    # Update Session Stats
    st.session_state.vote_count += 1
    if winner_source in st.session_state.session_wins:
        st.session_state.session_wins[winner_source] += 1
        
    # Log detailed session history
    st.session_state.session_history.append({
        "Round": st.session_state.vote_count,
        "Language": st.session_state.current_language,
        "Winner": winner_source,
        "Model A": st.session_state.model_a_name.split('/')[-1],
        "Model B": st.session_state.model_b_name.split('/')[-1]
    })

# --- VIEWS (ARENA VS STATISTICS) ---

def render_statistics_view():
    """Renders the Administrator/Global Statistics Page"""
    st.title("📊 Global Analytics Dashboard")
    st.markdown("Detailed aggregated statistics from all sessions.")
    
    if not os.path.exists(RESULTS_FILE):
        st.warning("No data available yet.")
        return

    df = pd.read_csv(RESULTS_FILE)
    if df.empty:
        st.warning("Dataset is empty.")
        return
        
    # Top Level Metrics
    total_votes = len(df)
    valid_votes = df[df['Winner_Source'] != 'Tie']
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Interactions", total_votes)
    
    if not valid_votes.empty:
        ms_wins = len(valid_votes[valid_votes['Winner_Source'] == 'MultiSynt'])
        hplt_wins = len(valid_votes[valid_votes['Winner_Source'] == 'HPLT'])
        ms_rate = round((ms_wins / len(valid_votes)) * 100, 1)
        hplt_rate = round((hplt_wins / len(valid_votes)) * 100, 1)
        
        c2.metric("MultiSynt Win Rate", f"{ms_rate}%", f"{ms_wins} wins")
        c3.metric("HPLT Win Rate", f"{hplt_rate}%", f"{hplt_wins} wins")
    
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌍 By Language", "🏆 Model Leaderboard", "📈 Trends", "💾 Raw Data"])
    
    with tab1:
        st.subheader("Win Rate by Language")
        # Prepare data for stacked bar chart
        lang_groups = df.groupby(['Language', 'Winner_Source']).size().unstack(fill_value=0)
        # Calculate percentages
        lang_pct = lang_groups.div(lang_groups.sum(axis=1), axis=0) * 100
        
        st.bar_chart(lang_groups)
        st.caption("Vote distribution per language.")
        
        with st.expander("View Percentage Table"):
            st.dataframe(lang_pct.style.format("{:.1f}%"))

    with tab2:
        st.subheader("Head-to-Head Performance")
        # Extract short names
        df['Model_A_Short'] = df['Model_A_Name'].apply(lambda x: x.split('/')[-1])
        
        # Filter where MultiSynt won
        ms_wins = df[df['Winner_Source'] == 'MultiSynt']
        if not ms_wins.empty:
            st.markdown("**Top Performing MultiSynt Models**")
            win_counts = ms_wins['Model_A_Short'].value_counts().reset_index()
            win_counts.columns = ['Model', 'Wins']
            st.dataframe(win_counts, use_container_width=True)
        else:
            st.info("No MultiSynt wins yet.")

    with tab3:
        st.subheader("Votes Over Time")
        # Convert timestamp
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Date'] = df['Timestamp'].dt.date
            daily_counts = df.groupby('Date').size()
            st.line_chart(daily_counts)
        except Exception as e:
            st.warning("Could not parse timestamps for trend analysis.")

    with tab4:
        st.subheader("Raw Data Inspector")
        st.dataframe(df)


def render_arena_view():
    """Renders the main voting arena"""
    st.title("⚔️ OELLM Arena")
    st.markdown(f"**Current Language:** {st.session_state.current_language}")

    # --- 1. DISCLAIMER ---
    with st.expander("ℹ️ **READ THIS FIRST: How to prompt Base Models**", expanded=True):
        st.markdown("""
        **These are Monolingual Base Models, not Chatbots.**
        * 🚫 **Don't ask questions** (e.g., *"What is the capital of Sweden?"*).
        * ✅ **DO write the start of a sentence** (e.g., *"The capital of Sweden is Stockholm, which is famous for..."*).
        * **Think "Auto-Complete":** You start the story, the AI finishes it.
        
        ---
        **Evaluation Guidelines:**
        * **Fluency:** Natural flow, correct grammar, and idiomatic usage.
        * **Bias:** To prevent bias, **model identities are hidden** and only revealed in aggregate every 5 votes.
        """)

    # --- 2. EXAMPLE PROMPTS ---
    st.markdown("### ✍️ Start writing or choose an example")
    cols = st.columns(3)
    example_list = EXAMPLE_PROMPTS.get(st.session_state.current_language, ["", "", ""])

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
                st.session_state.swap_models = random.choice([True, False])
                multisynt_options = MODELS_DB[st.session_state.current_language]['multisynt']
                chosen_multisynt = random.choice(multisynt_options)
                chosen_hplt = MODELS_DB[st.session_state.current_language]['hplt']
                
                try:
                    pipe_a = get_pipeline(chosen_multisynt, 0)
                    res_a = generate_text(pipe_a, user_prompt, 
                                        min_new_tokens=st.session_state.min_tokens,
                                        max_new_tokens=st.session_state.max_tokens,
                                        repetition_penalty=st.session_state.rep_penalty,
                                        temperature=st.session_state.temperature)
                    
                    pipe_b = get_pipeline(chosen_hplt, 1)
                    res_b = generate_text(pipe_b, user_prompt,
                                        min_new_tokens=st.session_state.min_tokens,
                                        max_new_tokens=st.session_state.max_tokens,
                                        repetition_penalty=st.session_state.rep_penalty,
                                        temperature=st.session_state.temperature)
                    
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
                winner = "HPLT" if st.session_state.swap_models else "MultiSynt"
                register_vote(winner)
                st.rerun()

        with col2:
            st.info(right_text)
            if st.button("Better Fluency (Model 2) 👉", key="vote_right", use_container_width=True):
                winner = "MultiSynt" if st.session_state.swap_models else "HPLT"
                register_vote(winner)
                st.rerun()
            
        if st.button("🤝 Tie / Equal Fluency", key="vote_tie", use_container_width=True):
            register_vote("Tie")
            st.rerun()

    # --- VOTE SUBMITTED & STATS ---
    if st.session_state.vote_submitted:
        vote_val = st.session_state.last_winner
        
        # Save logic
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
        
        # --- ANONYMIZED FEEDBACK LOGIC ---
        current_count = st.session_state.vote_count
        
        st.success("✅ Vote recorded successfully!")
        st.info("Identity hidden to prevent bias. Please start the next round.")

        # Check for Milestone (Every 5 votes)
        if current_count > 0 and current_count % 5 == 0:
            st.divider()
            st.markdown(f"### 📊 Session Milestone ({current_count} Votes)")
            st.write("Detailed breakdown of your session so far:")
            
            # 1. Summary Metrics
            wins = st.session_state.session_wins
            total = wins["MultiSynt"] + wins["HPLT"] + wins["Tie"]
            c1, c2, c3 = st.columns(3)
            c1.metric("MultiSynt Wins", wins["MultiSynt"])
            c2.metric("HPLT Wins", wins["HPLT"])
            c3.metric("Ties", wins["Tie"])
            
            # 2. Detailed History Table
            if st.session_state.session_history:
                st.markdown("#### recent Interactions")
                history_df = pd.DataFrame(st.session_state.session_history)
                st.dataframe(history_df.tail(10), use_container_width=True)
                
                # 3. Preference by Language (Session)
                st.markdown("#### Preference by Language (This Session)")
                session_grp = history_df.groupby(['Language', 'Winner']).size().unstack(fill_value=0)
                st.bar_chart(session_grp)

        else:
            remaining = 5 - (current_count % 5)
            st.caption(f"Vote count: {current_count}. Aggregate statistics will be revealed in {remaining} more votes.")

        st.divider()
        st.button("Start New Round", type="primary", on_click=reset_round)

# --- MAIN APP ROUTING ---

# Sidebar Header
if os.path.exists("oellm_logo.png"):
    st.sidebar.image("oellm_logo.png", width=120)
st.sidebar.markdown("A series of foundation models for transparent AI in Europe (https://openeurollm.eu/)")

if os.path.exists("airon_logo.png"):
    st.sidebar.image("airon_logo.png", width=200)
else:
    st.sidebar.markdown("**[Airon AI]**")
st.sidebar.markdown("Sponsor of [hardware and GPUs](https://www.airon.ai/).")
st.sidebar.divider()

# Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to:", ["⚔️ Arena", "📊 Analytics Dashboard"])
st.sidebar.divider()

if app_mode == "⚔️ Arena":
    # Sidebar config only for Arena
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("Choose a language to begin.")

    st.sidebar.selectbox(
        "Select Language", 
        sorted(list(MODELS_DB.keys())),
        key="current_language", 
        on_change=update_language 
    )

    st.sidebar.divider()

    with st.sidebar.expander("🛠️ Generation Settings", expanded=False):
        st.caption("Adjust these to ensure fair comparisons.")
        # Store in session state so they persist across re-runs
        if 'min_tokens' not in st.session_state: st.session_state.min_tokens = 70
        if 'max_tokens' not in st.session_state: st.session_state.max_tokens = 112
        if 'rep_penalty' not in st.session_state: st.session_state.rep_penalty = 1.2
        if 'temperature' not in st.session_state: st.session_state.temperature = 0.7

        st.session_state.min_tokens = st.slider("Min New Tokens", 10, 100, st.session_state.min_tokens)
        st.session_state.max_tokens = st.slider("Max New Tokens", 50, 512, st.session_state.max_tokens)
        st.session_state.rep_penalty = st.slider("Repetition Penalty", 1.0, 2.0, st.session_state.rep_penalty, step=0.05)
        st.session_state.temperature = st.slider("Temperature", 0.1, 1.5, st.session_state.temperature, step=0.1)

    render_arena_view()

elif app_mode == "📊 Analytics Dashboard":
    render_statistics_view()