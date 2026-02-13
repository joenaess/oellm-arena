import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_FILE = "arena_results.csv"
ASSETS_DIR = "assets"

def generate_plots():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found.")
        return

    df = pd.read_csv(RESULTS_FILE)
    if df.empty:
        print("Error: Dataset is empty.")
        return

    # Ensure assets directory exists
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # --- Plot 1: Win Rate by Language ---
    print("Generating Win Rate by Language plot...")
    plt.figure(figsize=(10, 6))
    
    # Filter for valid wins (excluding ties for win rate, or include them if preferred to match app)
    # The app code: lang_groups = df.groupby(["Language", "Winner_Source"]).size().unstack(fill_value=0)
    # It includes Ties. Let's do the same to match the app.
    
    lang_groups = df.groupby(["Language", "Winner_Source"]).size().unstack(fill_value=0)
    # Calculate percentages
    lang_pct = lang_groups.div(lang_groups.sum(axis=1), axis=0) * 100
    
    # Plotting
    ax = lang_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title("Win Rate by Language")
    plt.ylabel("Percentage")
    plt.xlabel("Language")
    plt.legend(title="Winner", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "win_rate_by_language.png"), dpi=300)
    plt.close()
    
    # --- Plot 2: Votes Over Time ---
    print("Generating Votes Over Time plot...")
    plt.figure(figsize=(12, 6))
    
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Date"] = df["Timestamp"].dt.date
        daily_counts = df.groupby("Date").size()
        
        daily_counts.plot(kind='line', marker='o', linestyle='-', color='b')
        plt.title("Votes Over Time")
        plt.ylabel("Number of Votes")
        plt.xlabel("Date")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(ASSETS_DIR, "votes_over_time.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating time series plot: {e}")

    print("Plots generated successfully in 'assets/' directory.")

if __name__ == "__main__":
    generate_plots()
