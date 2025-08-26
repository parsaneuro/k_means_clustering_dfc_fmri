import os
import numpy as np
import scipy.io
from tqdm import tqdm

# ===== USER PARAMETERS =====
mat_base = r'put the directory of your .mat files resulted from DynamicBC toolbox here'
npy_base = r'put the directory which you want to save .npy files and do the further analysis on them here'
groups = ['healthy', 'depressed']

# Specify ROI indices here (example: attention network)
node_indices = # choose nodes from you matrices which you want to explicitely do the further analysis on them


print(node_indices)
#"Sensory_Somatomotor_Hand": list(range(12, 41)),
#"Sensory_Somatomotor_Mouth": list(range(41, 46)),
#"Cingulo_opercular": list(range(46, 59)),
#"Auditory": list(range(60, 73)),
#"DMN": list(range(73, 83)) + list(range(85, 132)) + [136, 138],
#"Memory_Retrieval": list(range(132, 136)) + [220],
#"Visual": list(range(142, 172)),
#"Fronto_parietal": list(range(173, 182)) + list(range(185, 202)),
#"Salience": list(range(202, 220)),
#"Subcortical": list(range(221, 234)),
#"Ventral_Attention": [137, 234, 235, 236, 237, 238, 239, 240, 241],
#"Dorsal_Attention": [250, 251, 255, 256, 257, 258, 259, 260, 261, 262, 263],
#"attention" : [137, 234, 235, 236, 237, 238, 239, 240, 241,250, 251, 255, 256, 257, 258, 259, 260, 261, 262, 263]
#"Cerebellar": list(range(243, 247))

num_nodes = len(node_indices)
print(f"Extracting {num_nodes} ROIs: {node_indices}")

# === Number of subjects to process per group (set to None for all, or an integer) ===
max_subjects = 445  # Change this to the desired number, or None for all

def convert_group_mat_to_npy(mat_group_path, npy_group_path, max_subjects=None):
    os.makedirs(npy_group_path, exist_ok=True)
    subj_list = [s for s in os.listdir(mat_group_path) if os.path.isdir(os.path.join(mat_group_path, s))]
    subj_list = sorted(subj_list)  # Sort for reproducibility
    if max_subjects is not None:
        subj_list = subj_list[:max_subjects]
        print(f"Processing only the first {max_subjects} subjects out of {len(subj_list)} available.")

    for subj in tqdm(subj_list, desc=f"Subjects in {os.path.basename(mat_group_path)}"):
        subj_in = os.path.join(mat_group_path, subj)
        subj_out = os.path.join(npy_group_path, subj)
        os.makedirs(subj_out, exist_ok=True)
        window_files = [f for f in os.listdir(subj_in) if '_win' in f and f.endswith('.mat')]
        if not window_files:
            print(f"No window files found for {subj}")
            continue
        for wf in tqdm(window_files, leave=False, desc=f"{subj} windows"):
            mat_path = os.path.join(subj_in, wf)
            npy_path = os.path.join(subj_out, wf.replace('.mat', '.npy'))
            try:
                mat = scipy.io.loadmat(mat_path)
                # Find first array with correct shape (e.g., 264 x 264)
                matrix = next((val for val in mat.values() if isinstance(val, np.ndarray) and val.shape == (160, 160)), None)
                if matrix is None:
                    print(f"No valid 264x264 matrix found in {mat_path}, skipping.")
                    continue

                # Subset to your ROIs
                mat_roi = matrix[np.ix_(node_indices, node_indices)]

                # === CLIPPING: avoid exactly ±1, which cause inf in arctanh
                eps = 1e-7
                mat_roi = np.clip(mat_roi, -1 + eps, 1 - eps)

                # === FISHER Z-TRANSFORMATION
                mat_roi = np.arctanh(mat_roi)

                # === EXTRACT UPPER TRIANGLE (EXCLUDING DIAGONAL)
                triu = mat_roi[np.triu_indices(num_nodes, k=1)]

                # === ERROR CHECK: exclude files with NaNs or Infs after processing
                if np.isnan(triu).any() or np.isinf(triu).any():
                    print(f"NaNs or Infs after z-transform in {mat_path}, skipping.")
                    continue

                np.save(npy_path, triu)

            except Exception as e:
                print(f"Failed to convert {mat_path}: {e}")

for group in groups:
    mat_group_path = os.path.join(mat_base, group, 'FCM')
    npy_group_path = os.path.join(npy_base, group)
    print(f"\nConverting group '{group}': {mat_group_path} -> {npy_group_path}")
    convert_group_mat_to_npy(mat_group_path, npy_group_path, max_subjects=max_subjects)

print("\nDone! All .mat window files converted to .npy with Fisher z-normalization.")

