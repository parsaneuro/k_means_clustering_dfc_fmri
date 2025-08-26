# === MEMORY LEAK FIX: Set before numpy/sklearn imports ===
import os
os.environ['OMP_NUM_THREADS'] = '1'  # MUST be set before numpy/sklearn import

import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

# === USER PARAMETERS ===
npy_base = r'put the directory of the folder where results of npy_maker.py is saved there'
groups = ['healthy', 'depressed']
max_windows = 300000    # How many window vectors to use (subsample for speed)
Ks = list(range(2, 11))

# === LOAD RANDOM SUBSAMPLE OF WINDOW VECTORS ===
all_vecs = []
rng = np.random.default_rng(42)
for group in groups:
    group_dir = os.path.join(npy_base, group)
    for subj in tqdm(os.listdir(group_dir), desc=f"{group} subjects"):
        subj_dir = os.path.join(group_dir, subj)
        win_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.npy')])
        for wf in win_files:
            arr = np.load(os.path.join(subj_dir, wf))
            all_vecs.append(arr)
all_vecs = np.stack(all_vecs)
print(f"Total window vectors: {all_vecs.shape}")

# Subsample if necessary
if all_vecs.shape[0] > max_windows:
    idx = rng.choice(all_vecs.shape[0], size=max_windows, replace=False)
    vecs_sub = all_vecs[idx]
else:
    vecs_sub = all_vecs
print(f"Running elbow on {vecs_sub.shape[0]} window vectors")

# === ELBOW ANALYSIS ===
inertias = []
for K in Ks:
    print(f"Fitting KMeans: K={K}")
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        n_init=10,
        batch_size=256,    # You can increase this if memory allows, or leave as is
        random_state=42
    )
    kmeans.fit(vecs_sub)
    inertias.append(kmeans.inertia_)

plt.figure()
plt.plot(Ks, inertias, 'o-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('K-Means Elbow Plot')
plt.grid(True)
plt.show()



from sklearn.metrics import silhouette_score

# === SILHOUETTE SCORE ANALYSIS ===
silhouette_scores = []
for K in Ks:
    if K == 1:
        silhouette_scores.append(np.nan)
        continue
    print(f"Computing silhouette score for K={K}")
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        n_init=10,
        batch_size=256,
        random_state=42
    )
    labels = kmeans.fit_predict(vecs_sub)
    score = silhouette_score(vecs_sub, labels, metric='euclidean')
    silhouette_scores.append(score)

# === PLOT SILHOUETTE SCORES ===
plt.figure()
plt.plot(Ks, silhouette_scores, 'o-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.grid(True)
plt.show()
