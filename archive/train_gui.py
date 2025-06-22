import tkinter as tk
from tkinter import ttk, messagebox
import subprocess

# --- GUI Options ---
CHUNK_OPTIONS = [200, 300, 500, 1000]
PCA_OPTIONS = [20, 30, 50, 100]
TEST_SIZE_OPTIONS = [0.2, 0.3, 0.4, 0.5, 0.6]
LEARNING_RATE_OPTIONS = [0.01, 0.1, 0.2, 0.3, 0.5]
MAX_DEPTH_OPTIONS = [3, 5, 7, 9]
ESTIMATOR_OPTIONS = [100, 300, 500, 1000]
FEATURES = ['packet_length', 'rssi', 'snr']


def run_training():
    selected_features = [FEATURES[i] for i in feature_listbox.curselection()]
    if not selected_features:
        messagebox.showwarning("No Features Selected", "Please select at least one feature column.")
        return

    cmd = [
        "python", "train_core.py",
        "--chunk_size", str(chunk_var.get()),
        "--pca_components", str(pca_var.get()),
        "--test_size", str(test_var.get()),
        "--learning_rate", str(lr_var.get()),
        "--max_depth", str(depth_var.get()),
        "--n_estimators", str(estimators_var.get()),
        "--features", ','.join(selected_features),
        "--enable_confusion_matrix",
        "--enable_csv_log",
        "--enable_tensorboard"
    ]

    if grid_search_var.get():
        cmd.append("--use_grid_search")

    subprocess.run(cmd)


root = tk.Tk()
root.title("ML Training Config")
root.geometry("500x600")

# --- Dropdowns ---
ttk.Label(root, text="Chunk Size").pack()
chunk_var = tk.IntVar(value=300)
ttk.OptionMenu(root, chunk_var, 300, *CHUNK_OPTIONS).pack()

ttk.Label(root, text="PCA Components").pack()
pca_var = tk.IntVar(value=50)
ttk.OptionMenu(root, pca_var, 50, *PCA_OPTIONS).pack()

ttk.Label(root, text="Test Size").pack()
test_var = tk.DoubleVar(value=0.3)
ttk.OptionMenu(root, test_var, 0.3, *TEST_SIZE_OPTIONS).pack()

ttk.Label(root, text="Learning Rate").pack()
lr_var = tk.DoubleVar(value=0.1)
ttk.OptionMenu(root, lr_var, 0.1, *LEARNING_RATE_OPTIONS).pack()

ttk.Label(root, text="Max Depth").pack()
depth_var = tk.IntVar(value=7)
ttk.OptionMenu(root, depth_var, 7, *MAX_DEPTH_OPTIONS).pack()

ttk.Label(root, text="N Estimators").pack()
estimators_var = tk.IntVar(value=1000)
ttk.OptionMenu(root, estimators_var, 1000, *ESTIMATOR_OPTIONS).pack()

# --- Feature Selection Listbox ---
ttk.Label(root, text="Select Feature Columns").pack(pady=(10, 0))
feature_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, height=5, exportselection=0)
for feature in FEATURES:
    feature_listbox.insert(tk.END, feature)
feature_listbox.pack()

# --- Grid Search Checkbox ---
grid_search_var = tk.BooleanVar()
ttk.Checkbutton(root, text="Use Grid Search", variable=grid_search_var).pack(pady=10)

# --- Run Button ---
ttk.Button(root, text="Run Training", command=run_training).pack(pady=20)

root.mainloop()
