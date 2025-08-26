import os
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind, entropy as entropy_fn
from statsmodels.stats.multitest import fdrcorrection
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# === USER PARAMETERS ===
npy_base = r'put the directory of .npy files which you made them using npy_maker.py file in here'
groups = ['healthy', 'depressed']
k = 4 # Set your optimal K here!

# === Load data ===
subject_window_paths = []
subject_group_labels = []
subject_names = []

for group in groups:
    group_dir = os.path.join(npy_base, group)
    subjs = sorted([s for s in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, s))])
    for subj in subjs:
        subj_dir = os.path.join(group_dir, subj)
        win_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.npy')])
        subject_window_paths.append([os.path.join(subj_dir, wf) for wf in win_files])
        subject_group_labels.append(group)
        subject_names.append(subj)

all_subjects_vecs = []
good_subject_group_labels = []
good_subject_names = []
excluded_subjects = []

for win_paths, group, subj in tqdm(zip(subject_window_paths, subject_group_labels, subject_names), desc="Loading Subjects", total=len(subject_window_paths)):
    subj_vecs = []
    nan_found = False
    for wp in win_paths:
        arr = np.load(wp).astype(np.float32)
        if np.isnan(arr).any():
            nan_found = True
            break
        subj_vecs.append(arr)
    if nan_found or len(subj_vecs) == 0:
        excluded_subjects.append(subj)
        continue
    all_subjects_vecs.append(np.stack(subj_vecs))
    good_subject_group_labels.append(group)
    good_subject_names.append(subj)

if len(all_subjects_vecs) == 0:
    raise RuntimeError("No usable subjects found.")

print(f"Excluded {len(excluded_subjects)} subjects: {excluded_subjects}")
print(f"Remaining: {len(all_subjects_vecs)} subjects")

all_subjects_vecs = np.stack(all_subjects_vecs)
group_labels = np.array([0 if g == 'healthy' else 1 for g in good_subject_group_labels])
n_subjects, n_windows, n_features = all_subjects_vecs.shape

# === Extract exemplars: local peaks in variance over time ===
all_exemplars = []
for subj_data in all_subjects_vecs:
    variance_per_window = np.var(subj_data, axis=1)
    peaks, _ = find_peaks(variance_per_window)
    exemplars = subj_data[peaks]
    all_exemplars.append(exemplars)

all_exemplars = np.vstack(all_exemplars)

# === First K-means: on exemplars ===
kmeans_init = KMeans(n_clusters=k, n_init=500, random_state=42)
kmeans_init.fit(all_exemplars)
initial_centroids = kmeans_init.cluster_centers_

# === Final K-means: on full dataset using exemplar centroids ===
flat_data = all_subjects_vecs.reshape(-1, n_features)
kmeans = KMeans(n_clusters=k, init=initial_centroids, n_init=1, max_iter=500)
labels = kmeans.fit_predict(flat_data)
labels_reshaped = labels.reshape(n_subjects, n_windows)


# === Step 6: Compute State Metrics ===
def compute_metrics(seq, k):
    occ = [np.mean(seq == i) for i in range(k)]
    dwell = []
    for i in range(k):
        runs, r = [], 0
        for s in seq:
            if s == i: r += 1
            elif r > 0: runs.append(r); r = 0
        if r > 0: runs.append(r)
        dwell.append(np.mean(runs) if runs else 0)
    transitions = np.sum(np.diff(seq) != 0)
    counts = np.array([np.sum(seq == i) for i in range(k)])
    probs = counts / np.sum(counts)
    ent = entropy_fn(probs, base=2) if np.all(probs > 0) else 0
    return occ, dwell, transitions, ent

occ_all, dwell_all, trans_all, ent_all = [], [], [], []
for seq in labels_reshaped:
    occ, dwell, trans, ent = compute_metrics(seq, k)
    occ_all.append(occ)
    dwell_all.append(dwell)
    trans_all.append(trans)
    ent_all.append(ent)

occ_all = np.array(occ_all)
dwell_all = np.array(dwell_all)
trans_all = np.array(trans_all)
ent_all = np.array(ent_all)

import pandas as pd

# === Save transition counts per subject ===
transition_df = pd.DataFrame({
    'Subject': good_subject_names,
    'Group': good_subject_group_labels,
    'Transitions': trans_all
})

transition_df.to_excel('transitions_per_subject.xlsx', index=False)
print("Saved transitions to 'transitions_per_subject.xlsx'")


# === Step 7: Group Comparisons ===
def test_metric(metric):
    if metric.ndim == 1:
        _, p = ttest_ind(metric[group_labels == 0], metric[group_labels == 1])
        return np.array([p])
    else:
        return np.array([ttest_ind(metric[group_labels == 0, i], metric[group_labels == 1, i]).pvalue for i in range(metric.shape[1])])

def fdr_report(pvals, name):
    rejected, pvals_fdr = fdrcorrection(pvals, alpha=0.05)
    for i, (rej, pfdr) in enumerate(zip(rejected, pvals_fdr)):
        if rej:
            print(f"Significant group difference in {name} state {i} (FDR p={pfdr:.4g})")
    return rejected, pvals_fdr

p_occ = test_metric(occ_all)
p_dwell = test_metric(dwell_all)
p_trans = test_metric(trans_all)
p_ent = test_metric(ent_all)
rej_occ, pfdr_occ = fdr_report(p_occ, "occupancy")
rej_dwell, pfdr_dwell = fdr_report(p_dwell, "dwell")
print("Transitions p:", p_trans)
print("Entropy p:", p_ent)

# === Step 8: Print Metrics ===
def print_group_stats(metric_all, name):
    print(f"\n=== {name} ===")
    for i in range(k):
        h = metric_all[group_labels == 0, i]
        d = metric_all[group_labels == 1, i]
        print(f"State {i} - Healthy: {np.mean(h):.3f} ± {np.std(h):.3f}, Depressed: {np.mean(d):.3f} ± {np.std(d):.3f}")

print_group_stats(occ_all, "Occupancy")
print_group_stats(dwell_all, "Dwell Time")

# === Step 9: Visualization of p-values ===
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].bar(range(k), pfdr_occ)
axs[0, 0].set_title("Occupancy FDR-corrected p-values")
axs[0, 1].bar(range(k), pfdr_dwell)
axs[0, 1].set_title("Dwell Time FDR-corrected p-values")
axs[1, 0].bar([0], p_trans)
axs[1, 0].set_title("Transitions p-value")
axs[1, 1].bar([0], p_ent)
axs[1, 1].set_title("Entropy p-value")
for ax in axs.flat:
    ax.axhline(0.05, linestyle='--', color='red')
    ax.set_ylim(0, 1)
plt.suptitle("Group Comparison Metrics")
plt.tight_layout()
plt.show()

# === Step 10: Visualize Centroids ===
centroids = kmeans.cluster_centers_
num_nodes = int((1 + np.sqrt(1 + 8 * centroids.shape[1])) / 2)
fig, axs = plt.subplots(1, k, figsize=(4 * k, 4))
for i in range(k):
    mat = np.zeros((num_nodes, num_nodes))
    triu_ix = np.triu_indices(num_nodes, k=1)
    mat[triu_ix] = centroids[i]
    mat += mat.T
    axs[i].imshow(mat, cmap='bwr', vmin=-1, vmax=1)
    axs[i].set_title(f"State {i}")
    axs[i].axis('off')
plt.suptitle("DFC State Centroids")
plt.tight_layout()
plt.show()

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection

state_of_interest = 2  # Replace with your significant state index
n_nodes = num_nodes  # From your centroid plot step
n_edges = int(n_nodes * (n_nodes - 1) / 2)

# Prepare matrices
masks = labels_reshaped == state_of_interest
subject_mats = np.zeros((n_subjects, n_edges))  # Each row = subject, each col = edge

for subj_idx in range(n_subjects):
    state_wins = all_subjects_vecs[subj_idx][masks[subj_idx]]
    if len(state_wins) == 0:
        subject_mats[subj_idx] = np.nan
        continue
    mean_vec = np.mean(state_wins, axis=0)
    subject_mats[subj_idx] = mean_vec

# Remove subjects with no data in this state
valid_mask = ~np.isnan(subject_mats).any(axis=1)
subject_mats = subject_mats[valid_mask]
group_labels_valid = group_labels[valid_mask]

# Run edgewise t-tests
tvals, pvals = [], []
for edge_idx in range(n_edges):
    mdd_vals = subject_mats[group_labels_valid == 1, edge_idx]
    hc_vals = subject_mats[group_labels_valid == 0, edge_idx]
    tval, pval = ttest_ind(mdd_vals, hc_vals, equal_var=False)
    tvals.append(tval)
    pvals.append(pval)

# FDR correction
rejected, pvals_fdr = fdrcorrection(pvals, alpha=0.05)

import pandas as pd

# Initialize list for storing significant edges
sig_edges = []

for idx, (i, j) in enumerate(zip(*triu_ix)):
    if rejected[idx]:
        hc_vals = subject_mats[group_labels_valid == 0, idx]
        mdd_vals = subject_mats[group_labels_valid == 1, idx]
        hc_mean = np.mean(hc_vals)
        mdd_mean = np.mean(mdd_vals)
        diff = mdd_mean - hc_mean
        pfdr = pvals_fdr[idx]
        sig_edges.append({
            'Node1': i,
            'Node2': j,
            'HC_Mean': hc_mean,
            'MDD_Mean': mdd_mean,
            'Difference': diff,
            'FDR_p': pfdr
        })

# Convert list of significant edges to DataFrame
sig_edges_df = pd.DataFrame(sig_edges)

# Safety check: Only proceed if DataFrame is not empty
if not sig_edges_df.empty:
    # Optional: check what columns are available
    print("Columns in sig_edges_df:", sig_edges_df.columns.tolist())

    # Sort and save
    sig_edges_df = sig_edges_df.sort_values(by='FDR_p')
    sig_edges_df.to_excel(f'significant_edges_state_{state_of_interest}.xlsx', index=False)
    print(f"Saved significant edges to 'significant_edges_state_{state_of_interest}.xlsx'")

else:
    print(f"No significant edges found in state {state_of_interest}. No file saved.")



# Reconstruct significance matrix
sig_matrix = np.zeros((n_nodes, n_nodes))
diff_matrix = np.zeros((n_nodes, n_nodes))
triu_ix = np.triu_indices(n_nodes, k=1)

for idx, (i, j) in enumerate(zip(*triu_ix)):
    if rejected[idx]:
        sig_matrix[i, j] = 1
        sig_matrix[j, i] = 1
    # Group difference mean
    mdd_mean = np.mean(subject_mats[group_labels_valid == 1, idx])
    hc_mean = np.mean(subject_mats[group_labels_valid == 0, idx])
    diff_matrix[i, j] = mdd_mean - hc_mean
    diff_matrix[j, i] = diff_matrix[i, j]

# === Plot Difference Matrix with Significance Overlay ===
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 5))
plt.imshow(diff_matrix, cmap='bwr', vmin=-0.5, vmax=0.5)
plt.title(f'Difference Matrix (MDD - HC), State {state_of_interest}')
plt.colorbar(label='Connectivity Difference')
for i in range(n_nodes):
    for j in range(n_nodes):
        if sig_matrix[i, j]:
            plt.text(j, i, '*', ha='center', va='center', color='black')
plt.tight_layout()
plt.show()


# === Step 11: Participation Count ===
state_participation = np.zeros((k, 2))
for i in range(k):
    for g in [0, 1]:
        group_indices = np.where(group_labels == g)[0]
        count = sum([i in labels_reshaped[subj_idx] for subj_idx in group_indices])
        state_participation[i, g] = count

x = np.arange(1, k + 1)
bar_width = 0.35
plt.figure(figsize=(8, 5))
plt.bar(x - bar_width/2, state_participation[:, 0], bar_width, label='Healthy', color='gray')
plt.bar(x + bar_width/2, state_participation[:, 1], bar_width, label='Depressed', color='black')
plt.xlabel("State")
plt.ylabel("Subjects")
plt.title("State Participation by Group")
plt.xticks(x)
plt.legend()
plt.tight_layout()
plt.show()

# === Step 12: Transition Modeling Between Groups ===
# Compute subject-level transition matrices
transition_matrices = np.zeros((n_subjects, k, k))

for subj_idx, seq in enumerate(labels_reshaped):
    for t in range(len(seq) - 1):
        i, j = seq[t], seq[t + 1]
        transition_matrices[subj_idx, i, j] += 1
    # Normalize rows to get probabilities
    row_sums = transition_matrices[subj_idx].sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrices[subj_idx] /= row_sums

# Split matrices by group
trans_matrices_hc = transition_matrices[group_labels == 0]
trans_matrices_mdd = transition_matrices[group_labels == 1]

avg_trans_hc = np.mean(trans_matrices_hc, axis=0)
avg_trans_mdd = np.mean(trans_matrices_mdd, axis=0)
trans_diff = avg_trans_mdd - avg_trans_hc

# Build a DataFrame of transition differences
import pandas as pd

trans_data = []
for i in range(k):
    for j in range(k):
        trans_data.append({
            'From': i,
            'To': j,
            'HC_Prob': avg_trans_hc[i, j],
            'MDD_Prob': avg_trans_mdd[i, j],
            'Diff': trans_diff[i, j]
        })

trans_df = pd.DataFrame(trans_data)
trans_df = trans_df.sort_values(by='Diff', ascending=False)

# Print and save
print(trans_df.head(10))  # Most increased in MDD
print(trans_df.tail(10))  # Most decreased in MDD
trans_df.to_excel("transition_probabilities_comparison.xlsx", index=False)
print("Saved transition probabilities comparison to Excel.")

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 5))
plt.imshow(trans_diff, cmap='bwr', vmin=-0.2, vmax=0.2)
plt.colorbar(label='MDD - HC Transition Probability')
plt.title('Group Difference in Transition Probabilities')
plt.xlabel('To State')
plt.ylabel('From State')
plt.tight_layout()
plt.show()

# Flexibility: number of transitions
flexibility = [np.sum(np.diff(seq) != 0) for seq in labels_reshaped]
flexibility = np.array(flexibility)

print(f"Mean transitions - HC: {np.mean(flexibility[group_labels == 0]):.2f}")
print(f"Mean transitions - MDD: {np.mean(flexibility[group_labels == 1]):.2f}")

# Entropy of average transition matrices
def transition_entropy(matrix):
    flat = matrix.flatten()
    flat = flat[flat > 0]
    return -np.sum(flat * np.log2(flat))

entropy_hc = transition_entropy(avg_trans_hc)
entropy_mdd = transition_entropy(avg_trans_mdd)

print(f"Transition entropy - HC: {entropy_hc:.3f}")
print(f"Transition entropy - MDD: {entropy_mdd:.3f}")

# === Step 12b: Statistical Comparison of Individual Transitions ===

# Run t-tests on each transition (i→j) between groups
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection

transition_stats = []
pvals = []

for i in range(k):
    for j in range(k):
        hc_vals = trans_matrices_hc[:, i, j]
        mdd_vals = trans_matrices_mdd[:, i, j]
        t_stat, p_val = ttest_ind(mdd_vals, hc_vals, equal_var=False)
        pvals.append(p_val)
        transition_stats.append({
            'From': i,
            'To': j,
            'HC_Mean': np.mean(hc_vals),
            'MDD_Mean': np.mean(mdd_vals),
            'Difference': np.mean(mdd_vals) - np.mean(hc_vals),
            'p_value': p_val
        })

# FDR correction
rejected, pvals_fdr = fdrcorrection(pvals, alpha=0.05)
for idx, pfdr in enumerate(pvals_fdr):
    transition_stats[idx]['FDR_p'] = pfdr
    transition_stats[idx]['Significant'] = rejected[idx]

# Save to Excel
transition_stats_df = pd.DataFrame(transition_stats)
transition_stats_df = transition_stats_df.sort_values(by='FDR_p')
transition_stats_df.to_excel('significant_transition_differences.xlsx', index=False)
print("Saved transition significance comparisons to 'significant_transition_differences.xlsx'")

# Print significant transitions
print("\n=== Significant Transitions (FDR-corrected) ===")
for row in transition_stats_df[transition_stats_df['Significant'] == True].itertuples():
    print(f"Transition {row.From} → {row.To}: MDD={row.MDD_Mean:.3f}, HC={row.HC_Mean:.3f}, "
          f"Δ={row.Difference:.3f}, FDR_p={row.FDR_p:.4g}")


# === Step 13: Save Metrics and Statistical Results to Excel ===
import pandas as pd
from scipy.stats import ttest_ind
from numpy import mean, std

# --- A. Create subject-level metrics table ---
subject_data = {
    'Subject': good_subject_names,
    'Group': good_subject_group_labels,
    'Transitions': trans_all,
    'Entropy': ent_all
}

# Add occupancy and dwell time for each state
for i in range(k):
    subject_data[f'Occ_State_{i}'] = occ_all[:, i]
    subject_data[f'Dwell_State_{i}'] = dwell_all[:, i]

subject_df = pd.DataFrame(subject_data)
subject_df.to_excel('subject_level_metrics.xlsx', index=False)
print("Saved subject metrics to 'subject_level_metrics.xlsx'")

# --- B. Create statistical summary table ---
def compute_effect_size(group1, group2):
    # Cohen's d
    return (np.mean(group1) - np.mean(group2)) / np.sqrt(((np.std(group1)**2 + np.std(group2)**2) / 2))

stats_data = []

# Occupancy and Dwell Time
for metric_name, metric_all, pvals, pfdr in [
    ('Occupancy', occ_all, p_occ, pfdr_occ),
    ('DwellTime', dwell_all, p_dwell, pfdr_dwell)
]:
    for i in range(k):
        hc = metric_all[group_labels == 0, i]
        mdd = metric_all[group_labels == 1, i]
        effect = compute_effect_size(mdd, hc)
        stats_data.append({
            'Metric': metric_name,
            'State': i,
            'p_value': pvals[i],
            'p_FDR': pfdr[i],
            'Effect_Size': effect
        })

# Transitions and Entropy (1D metrics)
for metric_name, metric_all, pval in [
    ('Transitions', trans_all, p_trans[0]),
    ('Entropy', ent_all, p_ent[0])
]:
    hc = np.array(metric_all)[group_labels == 0]
    mdd = np.array(metric_all)[group_labels == 1]
    effect = compute_effect_size(mdd, hc)
    stats_data.append({
        'Metric': metric_name,
        'State': 'N/A',
        'p_value': pval,
        'p_FDR': 'N/A',
        'Effect_Size': effect
    })

# Save statistical results
stats_df = pd.DataFrame(stats_data)
stats_df.to_excel('group_comparison_statistics.xlsx', index=False)
print("Saved statistical results to 'group_comparison_statistics.xlsx'")

# === Step: Compute FC Strength Per Subject Per State ===
fc_strength_per_subject = np.zeros((n_subjects, k))
for subj_idx in range(n_subjects):
    for state in range(k):
        state_wins = all_subjects_vecs[subj_idx][labels_reshaped[subj_idx] == state]
        if len(state_wins) > 0:
            mean_vec = np.mean(state_wins, axis=0)
            fc_strength_per_subject[subj_idx, state] = np.mean(mean_vec)
        else:
            fc_strength_per_subject[subj_idx, state] = np.nan

# Save to Excel
group_names = ['Healthy' if g == 0 else 'Depressed' for g in group_labels]
fc_df = pd.DataFrame(fc_strength_per_subject, columns=[f'FC_Strength_State_{i}' for i in range(k)])
fc_df.insert(0, 'Group', group_names)
fc_df.insert(0, 'Subject', good_subject_names)
fc_df.to_excel('fc_strength_per_subject.xlsx', index=False)

# === Group Comparison ===
pvals = []
diffs = []
for i in range(k):
    hc_vals = fc_strength_per_subject[group_labels == 0, i]
    mdd_vals = fc_strength_per_subject[group_labels == 1, i]
    hc_vals = hc_vals[~np.isnan(hc_vals)]
    mdd_vals = mdd_vals[~np.isnan(mdd_vals)]
    t_stat, p_val = ttest_ind(mdd_vals, hc_vals, equal_var=False)
    pvals.append(p_val)
    diffs.append(np.mean(mdd_vals) - np.mean(hc_vals))

rejected, pvals_fdr = fdrcorrection(pvals, alpha=0.05)
stats_df = pd.DataFrame({
    'State': [f'State_{i}' for i in range(k)],
    'Difference (MDD - HC)': diffs,
    'p_value': pvals,
    'FDR_p_value': pvals_fdr,
    'Significant': rejected
})
stats_df.to_excel('fc_strength_group_comparison.xlsx', index=False)

# === NBS-style Comparison and Circular Plot ===
triu_ix = np.triu_indices(num_nodes, k=1)
n_edges = len(triu_ix[0])

for state in range(k):
    subject_edge_means = np.zeros((n_subjects, n_edges))
    for subj_idx in range(n_subjects):
        mask = labels_reshaped[subj_idx] == state
        if np.sum(mask) > 0:
            state_wins = all_subjects_vecs[subj_idx][mask]
            subject_edge_means[subj_idx] = np.mean(state_wins, axis=0)
        else:
            subject_edge_means[subj_idx] = np.nan

    valid_mask = ~np.isnan(subject_edge_means).any(axis=1)
    data = subject_edge_means[valid_mask]
    g_labels = group_labels[valid_mask]

    tvals, pvals, diffs = [], [], []
    for edge_idx in range(n_edges):
        hc_vals = data[g_labels == 0, edge_idx]
        mdd_vals = data[g_labels == 1, edge_idx]
        tval, pval = ttest_ind(mdd_vals, hc_vals, equal_var=False)
        tvals.append(tval)
        pvals.append(pval)
        diffs.append(np.mean(mdd_vals) - np.mean(hc_vals))

    rejected, pvals_fdr = fdrcorrection(pvals, alpha=0.05)

    sig_edges = []
    for idx, (i, j) in enumerate(zip(*triu_ix)):
        if rejected[idx]:
            sig_edges.append({
                'Node1': i,
                'Node2': j,
                'HC_Mean': np.mean(data[g_labels == 0, idx]),
                'MDD_Mean': np.mean(data[g_labels == 1, idx]),
                'Difference': diffs[idx],
                'FDR_p': pvals_fdr[idx]
            })

    sig_df = pd.DataFrame(sig_edges)
    if not sig_df.empty:
        fname = f'nbs_like_significant_edges_state_{state}.xlsx'
        sig_df.to_excel(fname, index=False)

        # === Circular Plot ===
        plt.figure(figsize=(8, 8))
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False).tolist()
        node_pos = np.array([(np.cos(a), np.sin(a)) for a in angles])

        for _, row in sig_df.iterrows():
            i, j = int(row['Node1']), int(row['Node2'])
            x = [node_pos[i][0], node_pos[j][0]]
            y = [node_pos[i][1], node_pos[j][1]]
            plt.plot(x, y, color='black', alpha=0.6)

        for idx, (x, y) in enumerate(node_pos):
            plt.scatter(x, y, s=100, c='red')
            plt.text(x * 1.1, y * 1.1, str(idx), ha='center', va='center')

        plt.title(f"Circular Plot of Significant Edges (State {state})")
        plt.axis('off')
        plt.savefig(f'circular_plot_state_{state}.png', dpi=300)
        plt.close()
    else:
        print(f"No significant edges in state {state}")


    # === Save Centroid State Matrices as CSV ===
import os

output_dir = 'centroid_matrices_csv'
os.makedirs(output_dir, exist_ok=True)

for idx, centroid in enumerate(centroids):
    mat = np.zeros((num_nodes, num_nodes))
    triu_ix = np.triu_indices(num_nodes, k=1)
    mat[triu_ix] = centroid
    mat += mat.T  # make symmetric
    df = pd.DataFrame(mat)
    csv_path = os.path.join(output_dir, f'centroid_state_{idx}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved centroid matrix for state {idx} to '{csv_path}'")


# === Step: Save Individual Transition Counts Per Subject ===
subject_transition_counts = []

for subj_idx, seq in enumerate(labels_reshaped):
    trans_matrix = np.zeros((k, k), dtype=int)
    for t in range(len(seq) - 1):
        i, j = seq[t], seq[t + 1]
        trans_matrix[i, j] += 1

    subj_data = {'Subject': good_subject_names[subj_idx],
                 'Group': good_subject_group_labels[subj_idx]}
    for i in range(k):
        for j in range(k):
            subj_data[f'Trans_{i}_to_{j}'] = trans_matrix[i, j]
    subject_transition_counts.append(subj_data)

# Create DataFrame
trans_counts_df = pd.DataFrame(subject_transition_counts)

# Save to Excel
trans_counts_df.to_excel('transition_counts_per_subject.xlsx', index=False)
print("Saved individual transition counts to 'transition_counts_per_subject.xlsx'")

