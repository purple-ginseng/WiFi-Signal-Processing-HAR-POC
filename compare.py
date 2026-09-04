"""Model Comparison: Tree-based vs Non-tree Models in Real-time BFM Motion Detection.

Evaluates and compares 6 models under Leave-One-Subject-Out (LOSO) Cross-Validation,
causal EMA smoothing, session-level leakage controls, and streaming latency benchmarks:

Tree-based Models:
  1. HistGradientBoostingClassifier (current deployed baseline from train_final.py)
  2. XGBoostClassifier (XGBClassifier)
  3. RandomForestClassifier (RandomForest)

Non-tree-based Models:
  4. LogisticRegression (StandardScaler + LogisticRegression from train_final.py)
  5. MLPClassifier (StandardScaler + Multi-Layer Perceptron Neural Network)
  6. SupportVectorClassifier (StandardScaler + Calibrated RBF-Kernel SVC)

All existing codebase files remain untouched.
"""

import os
import sys
import time
import json
import numpy as np

# Backward-compatibility layer for sessions_10hz.npz key ('fs' vs 'frequency')
_orig_np_load = np.load

class _CompatNpzWrapper:
    def __init__(self, npz):
        self._npz = npz

    def __getitem__(self, key):
        if key == 'frequency' and 'frequency' not in self._npz and 'fs' in self._npz:
            return self._npz['fs']
        return self._npz[key]

    def __contains__(self, key):
        return key in self._npz or (key == 'frequency' and 'fs' in self._npz)

    def __getattr__(self, name):
        return getattr(self._npz, name)

def _compat_np_load(*args, **kwargs):
    res = _orig_np_load(*args, **kwargs)
    if isinstance(res, np.lib.npyio.NpzFile):
        return _CompatNpzWrapper(res)
    return res

np.load = _compat_np_load

from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report)
import rtbfm


def ema_smooth(p, alpha=0.5):
    """Apply causal exponential moving average (EMA) smoothing over probabilities."""
    out = np.empty_like(p)
    acc = p[0]
    for i, v in enumerate(p):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def get_model_factories():
    """Return dictionary of factory functions instantiating each model."""
    return {
        # --- Tree-based Models ---
        'HistGradientBoosting': {
            'type': 'Tree-based',
            'factory': lambda: HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.08,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=0
            )
        },
        'XGBoost': {
            'type': 'Tree-based',
            'factory': lambda: XGBClassifier(
                n_estimators=250,
                learning_rate=0.08,
                max_depth=4,
                reg_lambda=1.0,
                random_state=0,
                eval_metric='logloss',
                n_jobs=-1
            )
        },
        'RandomForest': {
            'type': 'Tree-based',
            'factory': lambda: RandomForestClassifier(
                n_estimators=250,
                random_state=0,
                n_jobs=-1
            )
        },
        # --- Non-Tree-based Models ---
        'LogisticRegression': {
            'type': 'Non-tree: Linear',
            'factory': lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=0)
            )
        },
        'MLP (Neural Net)': {
            'type': 'Non-tree: Neural Net',
            'factory': lambda: make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=500,
                    early_stopping=True,
                    random_state=0
                )
            )
        },
        'SVC (RBF)': {
            'type': 'Non-tree: Kernel SVM',
            'factory': lambda: make_pipeline(
                StandardScaler(),
                CalibratedClassifierCV(
                    SVC(kernel='rbf', C=1.0),
                    ensemble=False
                )
            )
        }
    }


def evaluate_loso(model_name, model_info, X, y, session, subject, end_time):
    """Run Leave-One-Subject-Out Cross-Validation following train_final.py."""
    model_type = model_info['type']
    model_factory = model_info['factory']
    print(f"\n[{model_name}] ({model_type}) Starting Leave-One-Subject-Out (LOSO) Cross-Validation...")
    all_probs = np.zeros(len(y), dtype=np.float64)
    all_probs_smoothed = np.zeros(len(y), dtype=np.float64)
    unique_subjects = np.unique(subject)
    per_fold_bacc = {}
    per_fold_smoothed_bacc = {}
    fit_times = []

    t_cv_start = time.perf_counter()
    for subj in unique_subjects:
        test_mask = (subject == subj)
        model = model_factory()

        t0 = time.perf_counter()
        model.fit(X[~test_mask], y[~test_mask])
        fit_dur = time.perf_counter() - t0
        fit_times.append(fit_dur)

        probs = model.predict_proba(X[test_mask])[:, 1]
        all_probs[test_mask] = probs

        # Causal EMA smoothing per session
        for sid in np.unique(session[test_mask]):
            selection = np.where(test_mask)[0][session[test_mask] == sid]
            selection = selection[np.argsort(end_time[selection])]
            all_probs_smoothed[selection] = ema_smooth(all_probs[selection], alpha=0.5)

        fold_preds = (all_probs[test_mask] >= 0.5).astype(int)
        fold_smoothed_preds = (all_probs_smoothed[test_mask] >= 0.5).astype(int)
        fold_bacc = float(balanced_accuracy_score(y[test_mask], fold_preds))
        fold_sm_bacc = float(balanced_accuracy_score(y[test_mask], fold_smoothed_preds))
        per_fold_bacc[str(subj)] = fold_bacc
        per_fold_smoothed_bacc[str(subj)] = fold_sm_bacc
        print(f"  Fold '{subj}': fit={fit_dur:.2f}s | BAcc={fold_bacc:.4f} | Smoothed BAcc={fold_sm_bacc:.4f}")

    total_cv_time = time.perf_counter() - t_cv_start

    preds = (all_probs >= 0.5).astype(int)
    smoothed_preds = (all_probs_smoothed >= 0.5).astype(int)
    cm = confusion_matrix(y, preds)
    cm_smoothed = confusion_matrix(y, smoothed_preds)

    # Session-level accuracy: majority of window predictions match session ground truth
    sess_ok = [
        np.mean(preds[session == sid] == y[session == sid][0]) > 0.5
        for sid in np.unique(session)
    ]
    sess_ok_smoothed = [
        np.mean(smoothed_preds[session == sid] == y[session == sid][0]) > 0.5
        for sid in np.unique(session)
    ]

    report = classification_report(y, preds, target_names=['standing', 'walking'], digits=4, output_dict=True)
    report_smoothed = classification_report(y, smoothed_preds, target_names=['standing', 'walking'], digits=4, output_dict=True)

    metrics = {
        'model_family': model_type,
        'window_acc': float((preds == y).mean()),
        'window_balanced_acc': float(balanced_accuracy_score(y, preds)),
        'auc': float(roc_auc_score(y, all_probs)),
        'smoothed_acc': float((smoothed_preds == y).mean()),
        'smoothed_balanced_acc': float(balanced_accuracy_score(y, smoothed_preds)),
        'session_acc': float(np.mean(sess_ok)),
        'session_acc_smoothed': float(np.mean(sess_ok_smoothed)),
        'sessions_correct': f"{sum(sess_ok)}/{len(sess_ok)}",
        'confusion': cm.tolist(),
        'confusion_smoothed': cm_smoothed.tolist(),
        'per_fold_balanced_acc': per_fold_bacc,
        'per_fold_smoothed_balanced_acc': per_fold_smoothed_bacc,
        'cv_total_time_s': float(total_cv_time),
        'cv_mean_fit_time_s': float(np.mean(fit_times)),
        'standing_f1': float(report['standing']['f1-score']),
        'walking_f1': float(report['walking']['f1-score']),
        'macro_f1': float(report['macro avg']['f1-score']),
        'standing_recall': float(report['standing']['recall']),
        'walking_recall': float(report['walking']['recall']),
        'standing_precision': float(report['standing']['precision']),
        'walking_precision': float(report['walking']['precision']),
        'classification_report': report,
        'classification_report_smoothed': report_smoothed
    }
    return metrics, all_probs, all_probs_smoothed


def evaluate_permuted_control(model_factory, X, y, session, subject):
    """Run label-permutation control test (leakage check) across subjects."""
    rng = np.random.default_rng(0)
    y_copy = y.copy()
    sess = np.unique(session)
    label = {sid: y[session == sid][0] for sid in sess}
    perm = rng.permutation(list(label.values()))
    for sid, perm_label in zip(sess, perm):
        y_copy[session == sid] = perm_label

    results = []
    for subj in np.unique(subject):
        test_mask = (subject == subj)
        model = model_factory()
        model.fit(X[~test_mask], y_copy[~test_mask])
        pred = model.predict(X[test_mask])
        results.append(balanced_accuracy_score(y_copy[test_mask], pred))
    return float(np.mean(results))


def benchmark_latency(final_model, sessions, window_size, fixed_frequency, n_trials=500):
    """Benchmark end-to-end update latency and pure model predict latency."""
    s = sessions[0]
    W = int(round(window_size * fixed_frequency))

    # Pre-extract samples for pure model prediction latency
    features_list = []
    for k in range(n_trials):
        a = (k * 3) % (len(s['mag']) - W)
        v = rtbfm.window_features(s['mag'][a:a + W], s['phase'][a:a + W], fixed_frequency)
        features_list.append(v)

    # 1. Pure model inference time
    # Warmup
    for _ in range(20):
        final_model.predict_proba(features_list[0][None])

    t_model_start = time.perf_counter()
    for feat in features_list:
        final_model.predict_proba(feat[None])
    pure_model_ms = (time.perf_counter() - t_model_start) / n_trials * 1000.0

    # 2. End-to-end update time (features extraction + model inference)
    t_e2e_start = time.perf_counter()
    for k in range(n_trials):
        a = (k * 3) % (len(s['mag']) - W)
        v = rtbfm.window_features(s['mag'][a:a + W], s['phase'][a:a + W], fixed_frequency)
        final_model.predict_proba(v[None])
    e2e_ms = (time.perf_counter() - t_e2e_start) / n_trials * 1000.0

    return float(e2e_ms), float(pure_model_ms)


def main():
    print("=" * 105)
    print("                 REAL-TIME BFM MODEL COMPARISON: TREE VS NON-TREE")
    print("  HistGradientBoosting | XGBoost | RandomForest | LogisticRegression | MLP | SVC")
    print("=" * 105)

    window_size, hop_size = 2.0, 0.5
    print(f"\nLoading sessions (window={window_size}s, hop={hop_size}s)...")
    sessions, fixed_frequency = rtbfm.load_sessions()

    print(f"Building windows across {len(sessions)} sessions using parallel workers...")
    t0_data = time.perf_counter()
    outs = Parallel(n_jobs=-1)(
        delayed(rtbfm.build_windows)([s], fixed_frequency, window_size, hop_size)
        for s in sessions
    )
    X, y, session, env, subject, end_time = (np.concatenate([o[k] for o in outs]) for k in range(6))
    data_dur = time.perf_counter() - t0_data

    n_windows, n_features = X.shape
    walking_rate = float(y.mean())
    unique_subjects = sorted(list(np.unique(subject)))
    unique_sessions = sorted(list(np.unique(session)))

    print(f"Data prepared in {data_dur:.2f}s:")
    print(f"  Windows:      {n_windows}")
    print(f"  Features:     {n_features}")
    print(f"  Walking Rate: {walking_rate:.3f}")
    print(f"  Sessions:     {len(unique_sessions)}")
    print(f"  Subjects:     {unique_subjects}")

    factories = get_model_factories()
    comparison_summary = {
        'dataset': {
            'window_size_s': window_size,
            'hop_size_s': hop_size,
            'sampling_frequency_hz': fixed_frequency,
            'n_windows': n_windows,
            'n_features': n_features,
            'walking_rate': walking_rate,
            'n_sessions': len(unique_sessions),
            'subjects': unique_subjects
        },
        'models': {}
    }

    all_predictions = {}

    for model_name, info in factories.items():
        # 1. Leave-One-Subject-Out CV
        metrics, probs, probs_sm = evaluate_loso(
            model_name, info, X, y, session, subject, end_time
        )
        all_predictions[model_name] = {
            'raw_probs': probs,
            'smoothed_probs': probs_sm
        }

        # 2. Leakage control (Permuted label test)
        print(f"  Running permuted-label leakage control for {model_name}...")
        perm_bacc = evaluate_permuted_control(info['factory'], X, y, session, subject)
        metrics['permuted_label_control_bacc'] = perm_bacc
        print(f"  Permuted-label control BAcc (ideal ~0.500): {perm_bacc:.4f}")

        # 3. Final model fit and inference benchmark
        print(f"  Fitting final {model_name} model on all data and benchmarking latency...")
        final_model = info['factory']()
        final_model.fit(X, y)
        e2e_lat, pure_lat = benchmark_latency(final_model, sessions, window_size, fixed_frequency, n_trials=500)
        metrics['latency_e2e_per_update_ms'] = e2e_lat
        metrics['latency_model_inference_ms'] = pure_lat
        print(f"  End-to-end update latency: {e2e_lat:.3f} ms (budget = {hop_size * 1000:.0f} ms)")
        print(f"  Pure model inference:      {pure_lat:.3f} ms")

        comparison_summary['models'][model_name] = metrics

    # Save detailed results to JSON
    json_path = 'compare_results.json'
    with open(json_path, 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"\nSaved comprehensive comparison metrics to: {json_path}")

    # Display Consolidated Comparison Tables
    model_names = list(factories.keys())
    print("\n" + "=" * 115)
    print("                           CONSOLIDATED PERFORMANCE COMPARISON TABLE")
    print("=" * 115)
    header = f"{'Metric':<30} " + " ".join([f"{m[:13]:<13}" for m in model_names])
    print(header)
    print("-" * 115)

    def print_row(label, key, fmt="{:.4f}", percent=False):
        vals = []
        for m in model_names:
            v = comparison_summary['models'][m][key]
            if percent:
                vals.append(f"{v * 100:.2f}%")
            elif isinstance(v, (int, float)):
                vals.append(fmt.format(v))
            else:
                vals.append(str(v))
        row_str = f"{label:<30} " + " ".join([f"{val:<13}" for val in vals])
        print(row_str)

    print_row("Model Family", 'model_family')
    print("-" * 115)
    print_row("Per-window Accuracy", 'window_acc', percent=True)
    print_row("Per-window Balanced Acc", 'window_balanced_acc', percent=True)
    print_row("ROC AUC", 'auc', fmt="{:.4f}")
    print_row("Smoothed Acc (EMA)", 'smoothed_acc', percent=True)
    print_row("Smoothed Balanced Acc", 'smoothed_balanced_acc', percent=True)
    print_row("Session Accuracy", 'session_acc', percent=True)
    print_row("Sessions Correct (/50)", 'sessions_correct')
    print_row("Permuted Control (Leak)", 'permuted_label_control_bacc', fmt="{:.4f}")
    print_row("Standing F1-Score", 'standing_f1', fmt="{:.4f}")
    print_row("Walking F1-Score", 'walking_f1', fmt="{:.4f}")
    print_row("Macro Avg F1-Score", 'macro_f1', fmt="{:.4f}")
    print_row("Standing Recall (Acc)", 'standing_recall', percent=True)
    print_row("Walking Recall (Acc)", 'walking_recall', percent=True)
    print_row("CV Total Train Time", 'cv_total_time_s', fmt="{:.2f}s")
    print_row("Pure Model Inf (ms)", 'latency_model_inference_ms', fmt="{:.3f}ms")
    print_row("End-to-End Lat (ms)", 'latency_e2e_per_update_ms', fmt="{:.3f}ms")
    print("=" * 115)

    # Per-subject breakdown table
    print("\n" + "=" * 115)
    print("                  PER-SUBJECT BALANCED ACCURACY BREAKDOWN (LOSO CV)")
    print("                          Format: Raw / EMA-Smoothed")
    print("=" * 115)
    subj_header = f"{'Subject':<12} " + " ".join([f"{m[:15]:<16}" for m in model_names])
    print(subj_header)
    print("-" * 115)
    for subj in unique_subjects:
        parts = []
        for m in model_names:
            r = comparison_summary['models'][m]['per_fold_balanced_acc'][subj]
            s = comparison_summary['models'][m]['per_fold_smoothed_balanced_acc'][subj]
            parts.append(f"{r:.3f}/{s:.3f}")
        print(f"{subj:<12} " + " ".join([f"{p:<16}" for p in parts]))
    print("-" * 115)
    mean_parts = []
    for m in model_names:
        r = comparison_summary['models'][m]['window_balanced_acc']
        s = comparison_summary['models'][m]['smoothed_balanced_acc']
        mean_parts.append(f"{r:.3f}/{s:.3f}")
    print(f"{'MEAN':<12} " + " ".join([f"{p:<16}" for p in mean_parts]))
    print("=" * 115)

    print("\nComparison completed successfully!")


if __name__ == '__main__':
    main()
