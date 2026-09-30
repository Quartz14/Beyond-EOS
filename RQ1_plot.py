import torch
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import nltk 
import re 
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import random 
import os

random.seed(2025)

import time
start_time = time.perf_counter()


parser = argparse.ArgumentParser(description='get_gens')

parser.add_argument('--gpu', type=str, default='0', help='which gpu id to use')
parser.add_argument('--ckpts_list', type=str)
parser.add_argument('--noinst', type=str, default='', help='set this as _noinst to run the expts without system prompt')
parser.add_argument('--dataset', type=str, help='dataset name, typically on of [cactus, esconv, real_cbt, escot, escot_resp]')

args = parser.parse_args()
gpu_id = args.gpu.strip()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

ckpts = args.ckpts_list.split()
print("Working with ", ckpts)

files = ["L_0_5.pt", "L_0_10.pt" ,"L_0_15.pt" ,"L_0_20.pt","L_0_25.pt", "L_0_32.pt"]


def get_ckpt_gens(ckpt,dataset = args.dataset, files=files):
    text_dict = {}
    for f in files:
        x = torch.load(f"paper/ckpt_gens{args.noinst}/{dataset}/{ckpt}/{f}", weights_only=False)
        # text_dict[f] = x['texts'] #embeddings
        text_dict[f] = x['after_response']
    return text_dict



embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

p = pd.read_json("prototypes.json")
p = p.dropna()
print(p)
prototypes = p[args.dataset]
anchor_vectors = {}
for label, texts in prototypes.items():
    embeddings = embed_model.encode(texts, normalize_embeddings=True)
    # Mean pooling to get the single "Concept Vector"
    anchor_vectors[label] = np.mean(embeddings, axis=0).reshape(1, -1)

print("Probe calibrated. Anchors ready.")

def is_structural_noise(text, min_alphabet_ratio=0.4, min_unique_ratio=0.3):
    text = text.strip()
    if not text:
        return True 
    if re.search(r'\\x[0-9a-fA-F]{2}', text): return True
    if re.search(r'&#x[0-9a-fA-F]+', text): return True
    if re.search(r'(\W)\1{5,}', text): return True

    alpha_count = len(re.findall(r'[a-zA-Z]', text))
    total_count = len(text.replace(" ", ""))

    if total_count == 0: return True
    if (alpha_count / total_count) < min_alphabet_ratio:
        return True
    
    tokens = text.lower().split()
    if len(tokens) > 4: # Only check if sentence is long enough
        unique_tokens = set(tokens)
        diversity_ratio = len(unique_tokens) / len(tokens)

        if diversity_ratio < min_unique_ratio:
            return True
    
    return False


def classify_sentence(sentence):
    """
    Returns: 'Rule', 'Generation', or 'Noise'
    """
    # --- A. Structural Check for NOISE/REPETITION ---
    # 1. Length check: Single words or empty strings are noise
    sentence = sentence.strip()
    if len(sentence.split()) < 3: 
        return "Noise"
    
    # 2. Artifact check: repeated characters/garbage
    if is_structural_noise(sentence):
        return "Noise"
        
    # --- B. Semantic Check for RULE vs GEN ---
    # Embed the sentence
    sent_vec = embed_model.encode([sentence], normalize_embeddings=True)
    
    # Calculate distance to both Anchors
    sim_rule = cosine_similarity(sent_vec, anchor_vectors["Rule"])[0][0]
    sim_gen = cosine_similarity(sent_vec, anchor_vectors["Generation"])[0][0]
    # sim_cot = cosine_similarity(sent_vec, anchor_vectors["COT"])[0][0]
    
    # Assign to the closest anchor
    # if (sim_rule > sim_gen) and (sim_rule > sim_cot):
    #     return "Rule"
    # elif (sim_cot > sim_gen) and (sim_rule < sim_cot):
    #     return "COT"
    # else:
    #     return "Generation"


    if (sim_rule > sim_gen):# and (sim_rule > sim_cot):
        return "Rule"
    # elif (sim_cot > sim_gen) and (sim_rule < sim_cot):
    #     return "COT"
    else:
        return "Generation"


def sentence_tokenize(text):
    text = text.strip()
    blocks = re.split(r"\n+", text)
    final_sentences = []
    for block in blocks:
        block = block.strip()
        if not block: continue

        sub_blocks = re.split(r'\)\s*\(', block)

        for sub in sub_blocks:
            try:
                nltk_sents = nltk.sent_tokenize(sub)
            except:
                nltk_sents = [sub]

            final_sentences.extend(nltk_sents)
    return final_sentences

def analyze_generation_composition(generation_text):
    sents =  sentence_tokenize(generation_text)
    stats = {"Rule": 0, "Generation": 0, "COT":0, "Noise": 0, "Repetition": 0}
    total_words = 0
    seen_sentences = set()

    sentence_records = []

    for s in sents:
        word_count = len(s.split())
        if word_count == 0: continue
        total_words += word_count
        clean_s = s.strip()

        current_label = None
        if clean_s in seen_sentences:
            stats["Repetition"] +=word_count
            current_label = 'Repetition'

        else:
            label = classify_sentence(s)
            stats[label] += word_count 
            seen_sentences.add(clean_s)
            current_label = label

        sentence_records.append({
            "sentence": s, 
            "label": current_label,
            "word_count": word_count 
        })

    df_sentences = pd.DataFrame(sentence_records)
    composition = {}
    if total_words == 0:
        return {k:0.0 for k in stats},  pd.DataFrame(columns=["sentence", "label", "word_count"])
    
    for k,v in stats.items():
        composition[k] = v / total_words

    return composition, df_sentences

def aggregate_layer_statistics(generations_dict):
    layer_results = []
    layer_labels = []
    for layer_name, texts in generations_dict.items():
        layer_name, _ = layer_name.split(".")

        print(f"Processing layers {layer_name}")
        row_compostions = []
        all_sents = []
        for text in texts:
            comp, df_sent = analyze_generation_composition(text)
            row_compostions.append(comp)
            all_sents.append(df_sent)

        df_layer = pd.DataFrame(row_compostions)
        df_sents = pd.concat(all_sents)
        
        means = df_layer.mean().to_dict()

        means['Layer'] = layer_name
        layer_results.append(means)

        df_sents['layer'] = [layer_name] *len(df_sents)

        layer_labels.append(df_sents)

    return pd.DataFrame(layer_results), pd.concat(layer_labels)



final_stat = []
for ckpt in ckpts:
    text_dict = get_ckpt_gens(ckpt)
    final_stats_df, df_layer = aggregate_layer_statistics(text_dict)
    final_stats_df['ckpt'] = [ckpt]*len(final_stats_df)
    final_stat.append(final_stats_df)
    df_layer.to_csv(f"paper/ckpt_gens_labelled/{args.dataset}_{ckpt}{args.noinst}.csv",index=False)


stat_combined = pd.concat(final_stat, ignore_index=True)
stat_combined.to_csv(f"paper/ckpt_gens_labelled/{args.dataset}{args.noinst}_summary.csv",index=False)



def plot_stratified_stats(df, label_col='ckpt', data_name=args.dataset):
    unique_labels = df[label_col].unique()
    n_plots = len(unique_labels)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=True)
    if n_plots == 1:
        axes = [axes]
        
    # 3. Loop through labels and plot on corresponding axis
    for ax, label in zip(axes, unique_labels):
        # Filter data for this specific label
        subset = df[df[label_col] == label]
        
        # Plot lines
        ax.plot(subset['Layer'], subset['Rule'], marker="*", label='Rule')
        ax.plot(subset['Layer'], subset['Generation'], marker="o", label='Utterance')
        # ax.plot(subset['Layer'], subset['COT'], marker="o", label='COT')
        
        # Styling
        ax.set_title(f"Checkpoint Step: {label[4:]}", fontsize=14)
        ax.set_xlabel('Layer', fontsize=10)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
        
        if ax == axes[0]:
            ax.set_ylabel('Ratio in Generation', fontsize=10)

    plt.suptitle(f"Dataset: {data_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"paper/plots/{args.dataset}_ckpts{args.noinst}_layerwise.pdf", dpi=500)
    print("saved plot!")
    # plt.show()


final_stat = []
for ckpt in ckpts:
    print(ckpt)
    text_dict = get_ckpt_gens(ckpt)
    final_stats_df, df_layer = aggregate_layer_statistics(text_dict)
    final_stats_df['ckpt'] = [ckpt]*len(final_stats_df)
    final_stat.append(final_stats_df)
    df_layer.to_csv(f"paper/ckpt_gens_labelled/{args.dataset}_{ckpt}{args.noinst}.csv",index=False)
    # break


stat_combined = pd.concat(final_stat, ignore_index=True)
stat_combined.to_csv(f"paper/ckpt_gens_labelled/{args.dataset}{args.noinst}_summary.csv",index=False)

plot_stratified_stats(stat_combined)

end_time = time.perf_counter()

# Calculate the duration
duration = (end_time - start_time)/60

print(f"Code block took: {duration:.4f} minutes")
