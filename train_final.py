"""Train and honestly evaluate the deployed real-time detector.

Window = 2.0 s, hop = 0.5 s (see sweep_results.json: accuracy saturates by ~2 s,
longer windows buy <1 point of balanced accuracy for seconds of extra latency).
"""
import numpy as np
import json
import time
import os
from joblib import Parallel, delayed, dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
from sklearn.inspection import permutation_importance
import rtbfm

# MODEL_KIND picks the booster. Both are gradient-boosted trees on the same 43
# features; xgboost is the deployed one, hist is kept so the earlier numbers
# stay reproducible. Every fit in this file goes through make_model() so the
# LOSO folds, the permuted-label control and the final deployed fit can never
# drift apart.
MODEL_KIND = os.environ.get('MODEL_KIND', 'xgb')


def make_model():
    if MODEL_KIND == 'hist':
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                              max_leaf_nodes=15, l2_regularization=1.0,
                                              early_stopping=False, random_state=0)
    if MODEL_KIND == 'xgb':
        from xgboost import XGBClassifier
        # identical to the XGBoost entry in compare.py, so the model deployed
        # here is the one that model comparison actually measured
        return XGBClassifier(n_estimators=250, learning_rate=0.08, max_depth=4,
                             reg_lambda=1.0, random_state=0,
                             eval_metric='logloss', n_jobs=-1)
    raise ValueError('unknown MODEL_KIND %r' % MODEL_KIND)


# every artefact this script writes is tagged with the environment set and the
# booster, so an xgb run never silently overwrites a hist run's numbers
TAG = os.environ.get('MODEL_TAG', '_nofoil')
if MODEL_KIND != 'hist':
    TAG += '_' + MODEL_KIND


window_size, hop_size = 2.0, 0.5
sessions, fixed_frequency = rtbfm.load_sessions()

# Parallel is used to spread the build window tasks to all CPU core for faster processing
outs = Parallel(n_jobs=-1)(delayed(rtbfm.build_windows)([s], fixed_frequency, window_size, hop_size) for s in sessions)
X, y, session, env, subject, end_time = (np.concatenate([o[k] for o in outs]) for k in range(6))
print(f'{X.shape[0]} windows, {X.shape[1]} features, walking rate: {y.mean():.3f}')

# keep a part of weights from previous windows so that
# decision will not be easily affected by one noise window
def ema_smooth(p, alpha=0.5):
    out = np.empty_like(p); acc = p[0]
    for i, v in enumerate(p):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out

summary = {'window_size': window_size, 'hop_size': hop_size, 'n_windows': int(len(X)), 'n_features': int(X.shape[1])}

splits = [('leave-one-subject-out', subject)]
summary['n_sessions'] = int(len(np.unique(session)))
for split_name, folds in splits:
    all_probs = np.zeros(len(y))
    all_probs_smoothed = np.zeros(len(y))
    for f in np.unique(folds):
        test_mask = folds == f     # boolean mask for windows belonging to this subject
        model = make_model()
        model.fit(X[~test_mask], y[~test_mask]) # Use another 4 subjects as training set
        probs = model.predict_proba(X[test_mask])[:, 1]
        all_probs[test_mask] = probs
        # causal smoothing: EMA over each session's window stream (2 s window +
        # ~1 s of smoothing lag -> ~3 s end-to-end response time)
        for sid in np.unique(session[test_mask]):
            selection = np.where(test_mask)[0][session[test_mask] == sid]           # a mask that shows the test subject's windows belongs to the session id
            selection = selection[np.argsort(end_time[selection])]                  # reorder the windows in chronological order within the session based on end-time
            all_probs_smoothed[selection] = ema_smooth(all_probs[selection], 0.5)   # apply ema smoothing
    preds, smoothed_preds = (all_probs >= .5).astype(int), (all_probs_smoothed >= .5).astype(int)
    cm = confusion_matrix(y, preds)
    sess_ok = [np.mean(preds[session == sid] == y[session == sid][0]) > .5 for sid in np.unique(session)] # we only take the session with correct predictions more than 50%
    summary[split_name] = dict(
        window_acc=float((preds == y).mean()),
        window_balanced_acc=float(balanced_accuracy_score(y, preds)),
        auc=float(roc_auc_score(y, all_probs)),
        smoothed_acc=float((smoothed_preds == y).mean()),
        smoothed_balanced_acc=float(balanced_accuracy_score(y, smoothed_preds)),
        session_acc=float(np.mean(sess_ok)),
        confusion=cm.tolist(),
        per_fold_balanced_acc={str(f): float(balanced_accuracy_score(y[folds == f], preds[folds == f]))
                       for f in np.unique(folds)})
    print('\n== %s ==' % split_name)
    print(json.dumps({k: v for k, v in summary[split_name].items() if k != 'confusion'}, indent=1))
    print('confusion [rows=true standing/walking]:\n', cm)
    print(classification_report(y, preds, target_names=['standing', 'walking'], digits=3))
    if split_name == 'leave-one-subject-out':
        np.save('oof_proba%s.npy' % TAG, all_probs)
        np.save('oof_proba%s_smoothed.npy' % TAG, all_probs_smoothed)

# leakage control: same pipeline, labels permuted globally
rng = np.random.default_rng(0)
y_copy = y.copy()
sess = np.unique(session)
label = {sid: y[session == sid][0] for sid in sess}     # extract label for each session
perm = rng.permutation(list(label.values()))    # shuffle the labels
for sid, perm_label in zip(sess, perm):
    y_copy[session == sid] = perm_label
result = []
for subj in np.unique(subject):
    test_mask = subject == subj
    model = make_model()
    model.fit(X[~test_mask], y_copy[~test_mask])
    result.append(balanced_accuracy_score(y_copy[test_mask], model.predict(X[test_mask])))
summary['permuted_label_control_bacc'] = float(np.mean(result))
print('\npermuted-label control (should be ~0.5): %.3f' % np.mean(result))

# final model on everything
final_model = make_model()
final_model.fit(X, y)
dump({'model': final_model, 'window_s': window_size, 'hop_s': hop_size, 'fs': fixed_frequency, 'frequency': fixed_frequency,
      'environments': sorted(set(env.tolist())),
      'feature_names': rtbfm.feature_names(), 'classes': ['standing', 'walking']},
     'bfm_rt_model%s.joblib' % TAG)

# what is it actually using?
test_mask = subject == 'kenny'
model = make_model()
model.fit(X[~test_mask], y[~test_mask])
pi = permutation_importance(model, X[test_mask], y[test_mask], n_repeats=5, random_state=0,
                            scoring='balanced_accuracy', n_jobs=-1)
names = rtbfm.feature_names()
top = np.argsort(pi.importances_mean)[::-1][:12]
print('\ntop features (permutation importance, held-out subject):')
for i in top:
    print('   %-22s %.4f' % (names[i], pi.importances_mean[i]))
summary['top_features'] = [[names[i], float(pi.importances_mean[i])] for i in top]

# inference cost
s = sessions[0]
W = int(window_size * fixed_frequency)
t0 = time.perf_counter()
for k in range(500):
    a = (k * 3) % (len(s['mag']) - W)
    v = rtbfm.window_features(s['mag'][a:a+W], s['phase'][a:a+W], fixed_frequency)
    final_model.predict_proba(v[None])
dt = (time.perf_counter() - t0) / 500 * 1000
summary['per_update_ms'] = float(dt)
print('\nend-to-end per update (features + model): %.2f ms  (budget = %.0f ms hop)'
      % (dt, hop_size * 1000))
json.dump(summary, open('final_metrics%s.json' % TAG, 'w'), indent=1)
