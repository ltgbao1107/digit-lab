"""Digit Lab: run from the repository root with python train.py."""

import matplotlib
matplotlib.use("Agg")


# 1. Load the data
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

SEED = 42
OUT = Path('results')
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
digits = load_digits()
X, y = digits.data, digits.target
print(f'{len(y)} images | {X.shape[1]} pixels per image | {len(np.unique(y))} classes')

# 2. Reserve a test set
indices = np.arange(len(y))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
assert set(train_idx).isdisjoint(test_idx)
print(f'Training: {len(train_idx)} | Test: {len(test_idx)}')
fig, axes = plt.subplots(2, 5, figsize=(9, 4))
for digit, ax in enumerate(axes.flat):
    index = train_idx[np.flatnonzero(y_train == digit)[0]]
    ax.imshow(digits.images[index], cmap='gray_r', vmin=0, vmax=16)
    ax.set_title(f'Digit {digit}')
    ax.axis('off')
fig.suptitle('DIGIT LAB / One training example per class', fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'samples.png', bbox_inches='tight')
plt.show()
plt.close(fig)

# 3. Compare models without using test labels
models = {
    'Dummy baseline': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=SEED)),
    'RBF SVM': make_pipeline(StandardScaler(), SVC(C=3.0, kernel='rbf', gamma='scale')),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rows = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
    cv_rows.append({'model': name, 'cv_accuracy_mean': scores.mean(), 'cv_accuracy_std': scores.std()})
cv_table = pd.DataFrame(cv_rows)
selected_name = cv_table.loc[cv_table.cv_accuracy_mean.idxmax(), 'model']
print(cv_table.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
print('Selected before test evaluation:', selected_name)

# 4. Evaluate the fixed models
rows, predictions = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    rows.append({'model': name, 'test_accuracy': accuracy_score(y_test, pred),
                 'test_macro_f1': f1_score(y_test, pred, average='macro', zero_division=0),
                 'correct': int(np.sum(y_test == pred)), 'test_size': len(y_test)})
metrics = cv_table.merge(pd.DataFrame(rows), on='model')
metrics.to_csv(OUT / 'metrics.csv', index=False)
print(metrics.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
selected_model = models[selected_name]
selected_pred = predictions[selected_name]
report = classification_report(y_test, selected_pred, output_dict=True, zero_division=0)
(OUT / 'classification_report.json').write_text(json.dumps(report, indent=2))
pd.DataFrame({'dataset_index': test_idx, 'true_digit': y_test,
              'predicted_digit': selected_pred}).to_csv(OUT / 'predictions.csv', index=False)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(y_test, selected_pred, display_labels=np.arange(10), ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'{selected_name} / Held-out predictions', pad=16, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'confusion_matrix.png', bbox_inches='tight')
plt.show()
plt.close(fig)

# 5. Inspect mistakes
wrong = np.flatnonzero(selected_pred != y_test)
print(f'{selected_name}: {len(wrong)} mistakes out of {len(y_test)} test images')
fig, axes = plt.subplots(2, 6, figsize=(11, 4))
for ax in axes.flat:
    ax.axis('off')
for ax, position in zip(axes.flat, wrong[:12]):
    ax.imshow(X_test[position].reshape(8, 8), cmap='gray_r', vmin=0, vmax=16)
    ax.set_title(f'True {y_test[position]} / Pred {selected_pred[position]}', fontsize=9)
if not len(wrong):
    axes.flat[0].text(0.5, 0.5, 'No mistakes in this split', ha='center')
fig.suptitle('DIGIT LAB / Where the selected model fails', fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'mistakes.png', bbox_inches='tight')
plt.show()
plt.close(fig)

# 6. Stress-test with pixel noise
noise_rows = []
for sigma in [0, 1, 2, 4, 6]:
    accuracies = []
    for repeat in range(10):
        rng = np.random.default_rng(SEED + 100 + repeat)
        noisy = np.clip(X_test + rng.normal(0, sigma, X_test.shape), 0, 16)
        accuracies.append(accuracy_score(y_test, selected_model.predict(noisy)))
    noise_rows.append({'noise_sigma': sigma, 'accuracy_mean': float(np.mean(accuracies)),
                       'accuracy_std': float(np.std(accuracies)), 'repeats': 10})
noise_table = pd.DataFrame(noise_rows)
noise_table.to_csv(OUT / 'noise_robustness.csv', index=False)
print(noise_table.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(noise_table.noise_sigma, noise_table.accuracy_mean * 100,
            yerr=noise_table.accuracy_std * 100, fmt='o-', color='#2563eb', linewidth=2, capsize=5)
ax.set(xlabel='Gaussian noise standard deviation (pixel scale: 0–16)',
       ylabel='Test accuracy (%)', ylim=(0, 105), xticks=[0, 1, 2, 4, 6])
ax.grid(axis='y', alpha=0.2)
ax.set_title(f'{selected_name} / Accuracy under pixel noise', fontweight='bold', pad=14)
fig.text(0.5, 0.01, 'Error bars: standard deviation across 10 noise draws; not confidence intervals.', ha='center', fontsize=8)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(OUT / 'noise_robustness.png', bbox_inches='tight')
plt.show()
plt.close(fig)
metadata = {'seed': SEED, 'train_size': len(train_idx), 'test_size': len(test_idx),
            'selected_model': selected_name, 'selection_rule': 'highest 5-fold training CV accuracy',
            'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
            'train_indices': train_idx.tolist(), 'test_indices': test_idx.tolist()}
(OUT / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
print('Saved results to', OUT.resolve())
