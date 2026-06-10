"""
train_model.py  (v2 — Enhanced)
────────────────────────────────
Trains BOTH models for MedCore AI:
  1. Random Forest  (existing — retrained with cross-validation)
  2. ANN / Keras    (new — Artificial Neural Network with dropout)

Models saved to:
  models/heart_model.pkl  — RF for heart disease
  models/cancer_model.pkl — RF for cancer
  models/heart_ann.h5     — Keras ANN for heart disease
  models/cancer_ann.h5    — Keras ANN for cancer
  models/scaler.pkl       — StandardScaler (shared, for ANN inputs)

Run:
  python train_model.py
"""

print("🚀 MedCore AI — Model Training Script v2")
print("=" * 55)

import os, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings("ignore")

os.makedirs("models",   exist_ok=True)
os.makedirs("datasets", exist_ok=True)

# ─── Try importing TensorFlow ────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    print("✅ TensorFlow", tf.__version__, "available")
except ImportError:
    TF_AVAILABLE = False
    print("⚠ TensorFlow not installed — only Random Forest will be trained")
    print("   Install: pip install tensorflow")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — HEART DISEASE MODEL (Cleveland dataset)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("🫀  HEART DISEASE MODEL")
print("─" * 55)

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

heart = pd.read_csv("datasets/processed.cleveland.data", names=columns)
heart.replace("?", np.nan, inplace=True)
heart = heart.apply(pd.to_numeric)
heart.dropna(inplace=True)
heart["target"] = heart["target"].apply(lambda x: 1 if x > 0 else 0)

X_heart = heart.drop("target", axis=1)
y_heart  = heart["target"]

print(f"Dataset: {len(heart)} rows · {X_heart.shape[1]} features")
print(f"Class balance: No Disease = {(y_heart==0).sum()} | Disease = {(y_heart==1).sum()}")

X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
    X_heart, y_heart, test_size=0.2, random_state=42, stratify=y_heart
)

# ── 1a. Random Forest ────────────────────────────────────────────────────────
print("\n[RF] Training Random Forest …")

heart_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
heart_rf.fit(X_h_train, y_h_train)
y_h_pred = heart_rf.predict(X_h_test)
y_h_prob = heart_rf.predict_proba(X_h_test)[:, 1]

print(f"  Accuracy:  {accuracy_score(y_h_test, y_h_pred)*100:.1f}%")
print(f"  Precision: {precision_score(y_h_test, y_h_pred)*100:.1f}%")
print(f"  Recall:    {recall_score(y_h_test, y_h_pred)*100:.1f}%")
print(f"  AUC-ROC:   {roc_auc_score(y_h_test, y_h_prob):.3f}")

cv = cross_val_score(heart_rf, X_heart, y_heart, cv=StratifiedKFold(5), scoring="roc_auc")
print(f"  5-Fold CV AUC: {cv.mean():.3f} ± {cv.std():.3f}")

joblib.dump(heart_rf, "models/heart_model.pkl")
print("  ✅ Saved: models/heart_model.pkl")

# ── 1b. ANN (Keras) ──────────────────────────────────────────────────────────
if TF_AVAILABLE:
    print("\n[ANN] Training Heart Disease Neural Network …")

    scaler = StandardScaler()
    X_h_train_s = scaler.fit_transform(X_h_train)
    X_h_test_s  = scaler.transform(X_h_test)

    def build_ann(input_dim: int) -> tf.keras.Model:
        model = Sequential([
            # Input + first hidden layer
            Dense(128, activation="relu", input_dim=input_dim,
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            BatchNormalization(),
            Dropout(0.3),

            # Second hidden layer
            Dense(64, activation="relu",
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            BatchNormalization(),
            Dropout(0.3),

            # Third hidden layer
            Dense(32, activation="relu"),
            Dropout(0.2),

            # Output — sigmoid for binary classification
            Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        return model

    heart_ann = build_ann(X_h_train_s.shape[1])
    heart_ann.summary()

    callbacks = [
        EarlyStopping(monitor="val_auc", patience=15, restore_best_weights=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ]

    history_h = heart_ann.fit(
        X_h_train_s, y_h_train,
        validation_split=0.15,
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    _, acc_h, auc_h = heart_ann.evaluate(X_h_test_s, y_h_test, verbose=0)
    print(f"  ANN Accuracy: {acc_h*100:.1f}%")
    print(f"  ANN AUC-ROC:  {auc_h:.3f}")

    heart_ann.save("models/heart_ann.h5")
    joblib.dump(scaler, "models/scaler.pkl")
    print("  ✅ Saved: models/heart_ann.h5")
    print("  ✅ Saved: models/scaler.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CANCER MODEL (Wisconsin Breast Cancer)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("🔬  CANCER RISK MODEL")
print("─" * 55)

cancer = load_breast_cancer()
X_cancer = cancer.data[:, :10]   # first 10 features
y_cancer  = cancer.target         # 1 = benign, 0 = malignant

print(f"Dataset: {len(y_cancer)} rows · {X_cancer.shape[1]} features")
print(f"Class balance: Malignant = {(y_cancer==0).sum()} | Benign = {(y_cancer==1).sum()}")

X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)

# ── 2a. Random Forest ────────────────────────────────────────────────────────
print("\n[RF] Training Random Forest …")

cancer_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
cancer_rf.fit(X_c_train, y_c_train)
y_c_pred = cancer_rf.predict(X_c_test)
y_c_prob = cancer_rf.predict_proba(X_c_test)[:, 1]

print(f"  Accuracy:  {accuracy_score(y_c_test, y_c_pred)*100:.1f}%")
print(f"  Precision: {precision_score(y_c_test, y_c_pred)*100:.1f}%")
print(f"  Recall:    {recall_score(y_c_test, y_c_pred)*100:.1f}%")
print(f"  AUC-ROC:   {roc_auc_score(y_c_test, y_c_prob):.3f}")

cv_c = cross_val_score(cancer_rf, X_cancer, y_cancer, cv=StratifiedKFold(5), scoring="roc_auc")
print(f"  5-Fold CV AUC: {cv_c.mean():.3f} ± {cv_c.std():.3f}")

joblib.dump(cancer_rf, "models/cancer_model.pkl")
print("  ✅ Saved: models/cancer_model.pkl")

# ── 2b. ANN (Keras) ──────────────────────────────────────────────────────────
if TF_AVAILABLE:
    print("\n[ANN] Training Cancer Neural Network …")

    # Use same scaler (refit on cancer data)
    cancer_scaler = StandardScaler()
    X_c_train_s = cancer_scaler.fit_transform(X_c_train)
    X_c_test_s  = cancer_scaler.transform(X_c_test)

    cancer_ann = build_ann(X_c_train_s.shape[1])

    history_c = cancer_ann.fit(
        X_c_train_s, y_c_train,
        validation_split=0.15,
        epochs=150,
        batch_size=16,
        callbacks=callbacks,
        verbose=0,
    )

    _, acc_c, auc_c = cancer_ann.evaluate(X_c_test_s, y_c_test, verbose=0)
    print(f"  ANN Accuracy: {acc_c*100:.1f}%")
    print(f"  ANN AUC-ROC:  {auc_c:.3f}")

    cancer_ann.save("models/cancer_ann.h5")
    print("  ✅ Saved: models/cancer_ann.h5")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("✅  ALL MODELS TRAINED SUCCESSFULLY")
print("=" * 55)
print("""
Models saved to ./models/:
  heart_model.pkl   — Heart disease Random Forest
  cancer_model.pkl  — Cancer risk Random Forest
  scaler.pkl        — StandardScaler for ANN inputs
""")

if TF_AVAILABLE:
    print("""  heart_ann.h5      — Heart disease ANN (Keras)
  cancer_ann.h5     — Cancer risk ANN (Keras)

Ensemble prediction (RF + ANN) is available in ml/predict.py
""")
else:
    print("  (ANN models not trained — install TensorFlow to enable)\n")

print("Next steps:")
print("  1. streamlit run dashboard.py     # Launch Streamlit UI")
print("  2. python app.py                  # Launch Flask API (port 5000)")
print("  3. python database/db.py          # Initialise SQLite database")