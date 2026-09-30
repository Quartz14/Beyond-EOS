import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import nltk 
import re 
import matplotlib.pyplot as plt
import hdbscan
import argparse
import random 
import os
random.seed(2025)
from sklearn.metrics import silhouette_score
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.ticker as mtick
import seaborn as sns
sns.set_theme(style="whitegrid")

import time
start_time = time.perf_counter()


parser = argparse.ArgumentParser(description='get_gens')

parser.add_argument('--ckpt', type=str)
parser.add_argument('--dataset', type=str) #[cactus, esconv, real_cbt, escot, escot_response??]

args = parser.parse_args()
gpu_id = args.gpu.strip()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

gen = pd.read_csv(f"paper/ckpt_gens_labelled/{args.dataset}_{args.ckpt}.csv")
# sentence	label	word_count	layer

print("no. labelled as rules: ", len(gen[gen['label']=='Rule']))
print("no. labelled as generation: ", len(gen[gen['label']=='Generation']))
generations = list(gen[gen['label'].isin(['Rule', 'Generation'])]['sentence'])


embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

p = pd.read_json("prototypes.json")
prototypes = p[args.dataset]

anchor_vectors = {}
anchor_keys = []
anchor_matrix = []

print("Computing Anchor Centroids...")
for label, texts in prototypes.items():
    embeddings = embed_model.encode(texts, normalize_embeddings=True)
    centroid = np.mean(embeddings, axis=0)
    anchor_vectors[label] = centroid
    anchor_keys.append(label)
    anchor_matrix.append(centroid)

anchor_matrix = np.array(anchor_matrix)

print("Embedding Generations...")

generations_set = list(set(generations))

gen_vectors = embed_model.encode(generations_set, normalize_embeddings=True)

# Assign Labels based on Proximity to Centroids (Supervised)
# We compute Cosine Similarity to ALL anchors
similarities = cosine_similarity(gen_vectors, anchor_matrix) # Shape: (N, 5)

sorted_sims = np.sort(similarities, axis=1) # Sorts low -> high
best_score = sorted_sims[:, -1]   
second_best = sorted_sims[:, -2]
confidence_absolute = best_score
confidence_margin = best_score - second_best

closest_anchor_indices = np.argmax(similarities, axis=1)
max_scores = np.max(similarities, axis=1)
pred_labels = [anchor_keys[i] for i in closest_anchor_indices]

print("Running UMAP...")
reducer = umap.UMAP(n_neighbors=100, n_components=2, metric='cosine', random_state=42)
embedding_2d = reducer.fit_transform(gen_vectors)

# --- 4. HDBSCAN CLUSTERING (The "Discovery" Step) ---
# This finds the ACTUAL clusters in the UMAP space, ignoring your centroids.
print("Running HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=100)
found_clusters = clusterer.fit_predict(embedding_2d)

# --- 5. BUILD DATAFRAME & VISUALIZE ---
df = pd.DataFrame(embedding_2d, columns=['x', 'y'])
df['text'] = generations_set
df['Centroid_Label'] = pred_labels        
df['Confidence'] = max_scores
df['Found_Cluster'] = found_clusters    

df['Confidence_Abs'] = confidence_absolute
df['Confidence_Margin'] = confidence_margin

X_umap = df[['x', 'y']].values
labels = df['Centroid_Label'].values
score = silhouette_score(X_umap, labels)
print("SILHOUTTE SCORE = ", score)
if score > 0.5:
    print("Strong separation. The centroids align with the geometry.")
elif score > 0.2:
    print("Weak separation. There is some overlap/confusion.")
else:
    print("No separation. The centroids do not match the data shape.")


def extract_top_keywords(df, cluster_col='Found_Cluster', top_n=10):
    print(f"\n--- Interpreting Clusters from {cluster_col} ---")
    
    # Group all text by cluster
    docs_per_cluster = df.groupby(cluster_col)['text'].apply(lambda x: " ".join(x))
    
    # Use CountVectorizer to find frequent words (simple c-TF-IDF proxy)
    # stop_words='english' removes "the", "is", etc.
    count = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    count_matrix = count.fit_transform(docs_per_cluster)
    features = count.get_feature_names_out()
    
    for i, cluster_id in enumerate(docs_per_cluster.index):
        if cluster_id == -1: continue # Skip noise
        
        # Get top words for this cluster
        row = count_matrix[i].toarray().flatten()
        top_indices = row.argsort()[-top_n:][::-1]
        top_words = [features[idx] for idx in top_indices]
        
        print(f"Cluster {cluster_id}: {top_words}")


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df, 
    x='x', 
    y='y', 
    hue='Centroid_Label' 
)

plt.title("Generations Colored by Nearest Centroid (Theory)")
plt.show()

generations_set_rules = list(set(gen[gen['label'].isin(['Rule'])]['sentence']))

gen_vectors = embed_model.encode(generations_set_rules, normalize_embeddings=True)

reducer = umap.UMAP(n_neighbors=50, n_components=2, metric='cosine', random_state=42)
embedding_2d = reducer.fit_transform(gen_vectors)
print("Running HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=50)
found_clusters = clusterer.fit_predict(embedding_2d)


df_rules = pd.DataFrame(embedding_2d, columns=['x', 'y'])
df_rules['text'] = generations_set_rules
df_rules['Found_Cluster'] = found_clusters


X_umap = df_rules[['x', 'y']].values


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df_rules, 
    x='x', 
    y='y',
    hue=df_rules['Found_Cluster'].astype(str) 
)

plt.title("Rules Clustered by HDBSCAN (Structural Reality)")
plt.legend(title='Found_Cluster') 
plt.show()


extract_top_keywords(df_rules, 'Found_Cluster')

# Question = Across layer phases do the question cluster change?

# --- 1. Data Preparation (Same as your code) ---
composition = df_rules.groupby(['Found_Cluster', 'Layer_Range']).size().reset_index(name='count')
total_per_cluster = composition.groupby('Found_Cluster')['count'].transform('sum')
composition['percentage'] = composition['count'] / total_per_cluster
composition = composition[composition['Found_Cluster'] != -1]

# --- 2. Pivot Data for Stacked Plot ---
pivot_df = composition.pivot(
    index='Found_Cluster', 
    columns='Layer_Range', 
    values='percentage'
)

# --- 3. Enforce Sort Order ---
layer_order = ["0-5", "0-10", "0-15", "0-20", "0-25", "0-32"]
# Filter to only existing columns to avoid errors if some ranges are missing
existing_order = [l for l in layer_order if l in pivot_df.columns]
pivot_df = pivot_df[existing_order]

# --- 4. Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))

pivot_df.plot(
    kind='bar', 
    stacked=True, 
    ax=ax,
    width=0.8,
    # You can specify a colormap here if you want, e.g., colormap='viridis'
)

for c in ax.containers:
    
    # Optional: labels = [f'{v.get_height():.1%}' if v.get_height() > 0 else '' for v in c]
    # But explicitly looping allows better control over position and threshold
    
    for rect in c:
        height = rect.get_height()
        
        # Only print if the segment is large enough (e.g., > 2%)
        if height > 0.02: 
            # Calculate the center x and y for the text
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + height / 2
            
            ax.text(
                x, y, 
                f'{height:.0%}', # Format as percentage (e.g., 50%)
                ha='center', 
                va='center', 
                color='white', 
                fontsize=9,
                fontweight='bold'
            )

# --- 5. Formatting ---
ax.set_title("Provenance of Rule Types: Which Layers drive which behaviors?", fontsize=14, pad=20)
ax.set_xlabel("Rule Type (Found_Cluster)", fontsize=12)
ax.set_ylabel("Share of Cluster", fontsize=12)

# Format y-axis as percentages
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.legend(title='Layer_Range', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()