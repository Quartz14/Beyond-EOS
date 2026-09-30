import os
import torch 
import pandas as pd
import numpy as np

import re 

def extract_note(generations, model='llama'):

    refusal_pattern = r"([^.]*I'm sorry, but I can't provide[^.]*\.)"

    if(model=='llama'):
        pattern = r"\(Note:\s+(.*?)\)"

    elif(model =='qwen'):
        pattern = r"(?<=\n\n)\(.*?\)\s"

    elif(model =='gemma'):
        pattern = r"(\*\s\*\*.*?\.)"
        
    all_extracted_notes = {}
    all_extracted_refusals = {}

    for ri, text in enumerate(generations):
            matches = re.findall(pattern, text)
            refusals = re.findall(refusal_pattern, text, flags=re.IGNORECASE)
            all_extracted_notes[ri] = [r.strip() for r in matches]
            all_extracted_refusals[ri] = [r.strip() for r in refusals]

    all_extracted_notes = {
        key: set([
            fragment.strip()                      
            for sentence in value                
            for fragment in sentence.split('.')   
            if fragment.strip()                   
        ]) for key, value in all_extracted_notes.items()}

    return all_extracted_notes, all_extracted_refusals


def extract_notes_refusal(gemma_path='paper/ckpt_gens/cactus_gemma', model='llama'):
    gemma_dict = {}
    refusals = {}
    for file in os.listdir(gemma_path):
        file_path = os.path.join(gemma_path, file)
        ckpt_name = file.split(".")[0]
        ds = torch.load(file_path)
        gemma_notes, gemma_refusal = extract_note(list(ds['after_response']), model=model)
        gemma_dict[ckpt_name] = gemma_notes
        refusals[ckpt_name] = gemma_refusal

    return gemma_dict, refusals

llama_dict, llama_refusal = extract_notes_refusal('paper/ckpt_gens/cactus_llama',model='llama')
gemma_dict, gemma_refusal = extract_notes_refusal(model='gemma')
qwen_dict, qwen_refusal = extract_notes_refusal('paper/ckpt_gens/cactus_qwen', model='qwen')

all_model_data = {'llama': llama_dict,
             'qwen': qwen_dict,
             'gemma': gemma_dict}

import pandas as pd
import plotly.express as px

# 1. DATA TRANSFORMATION
# Convert your Dict[Model, Dict[Ckpt, Dict[Index, List]]] into a Flat DataFrame
records = []
# Assuming structure: all_models_data = {'Model A': {100: {0: [], 1: ['Note...']}}}
for model_name, checkpoints in all_model_data.items():
    for ckpt, indices in checkpoints.items():
        for idx, notes in indices.items():
            records.append({
                'Model': model_name,
                'Checkpoint': ckpt,
                'Test_Index': int(idx),
                'Has_Rule': len(notes) > 0,      # Binary Trigger
                'Rule_Volume': len(notes),       # Count
                'Content_Length': sum(len(n) for n in notes) # Intensity
            })

df = pd.DataFrame(records)
custom_order = ['ckpt0',  'ckpt50','ckpt600', 'ckpt1000']
                
df['Checkpoint'] = pd.Categorical(df['Checkpoint'], categories=custom_order, ordered=True)
# 3. Sort the DataFrame by the 'Size' column
df_sorted = df.sort_values(by=['Model','Checkpoint', 'Test_Index'])


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


target_checkpoints = ['ckpt0', 'ckpt50', 'ckpt600'] # Replace with your actual checkpoint names

# 2. Determine common color scaling (vmin/vmax)
# We filter the dataframe for these checkpoints to find the global min/max.
# Note: If your aggregation is 'sum' instead of 'mean', you might need to calculate the pivots first to get the true max.
subset_df = df[df['Checkpoint'].isin(target_checkpoints)]
vmin = 0  # Assuming fillna(0) sets the floor
vmax = subset_df['Rule_Volume'].max() # Use .max() of the column for 'mean' aggregation

# 3. Create the subplots (3 Rows, 1 Column)
fig, axes = plt.subplots(3, 1, figsize=(12, 5)) # Adjusted height for 3 rows

# 4. Iterate through checkpoints and axes
for i, (ckpt, ax) in enumerate(zip(target_checkpoints, axes)):
    
    # --- Data Preparation (Same as your snippet) ---
    df_trigger = df[df['Checkpoint'] == ckpt].pivot_table(
        index='Model', 
        columns='Test_Index', 
        values='Rule_Volume', 
        aggfunc='mean'
    ).fillna(0)

    # --- Plotting ---
    im = ax.imshow(
        df_trigger.values, 
        cmap='Greens', 
        aspect='auto', 
        interpolation='nearest',
        vmin=vmin,  # <--- Enforce common min
        vmax=vmax   # <--- Enforce common max
    )

    # --- Formatting (Applied to each subplot) ---
    ax.grid(False)
    tick_interval = 10
    ax.set_xticks(np.arange(0, len(df_trigger.columns), tick_interval))
    ax.set_xticklabels(df_trigger.columns[::tick_interval], rotation=90, fontsize=8)

    ax.set_yticks(np.arange(len(df_trigger.index)))
    ax.set_yticklabels(df_trigger.index, fontsize=10)
    
    # Label each row with its checkpoint
    ax.set_ylabel("Model", fontsize=10)
    ax.set_title(f"Checkpoint: {ckpt}", fontsize=10, pad=3, fontweight='bold')

    # Add the vertical lines and text annotations
    labels = [1, 3, 5, 7, 9, 11, 13]
    start_x = 29
    step = 30

    for i, ax in enumerate(axes):
        # Iterate through your labels/positions
        for j, label in enumerate(labels):
            x_pos = start_x + (j * step)
            
            # 1. Draw Line: executed for ALL plots
            ax.axvline(x=x_pos + 0.5, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
            
            # 2. Add Text: executed ONLY for the bottom plot
            if i == len(axes) - 1:
                ax.text(
                    x_pos + 0.55, -3, 
                    f"{label} turns", 
                    color='black', 
                    ha='right',
                    # rotation=90, 
                    va='top', 
                    fontsize=9, 
                    # fontweight='bold'
                )
        
        # 3. Add X-axis label: executed ONLY for the bottom plot
        if i == len(axes) - 1:
            ax.set_xlabel("Test Indices")

cbar = plt.colorbar(im, ax=ax, shrink=0.9, fraction=0.01, pad=0.01)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
plt.subplots_adjust(hspace=0.6) # Increase vertical space if text overlaps
plt.savefig("paper/plots/rq3_index_ckpt_presence_plot.pdf", format="pdf", dpi=500)
plt.show()
