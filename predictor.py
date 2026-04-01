"""
CodeCosh - ML Engine (predictor.py)
"Know Your Code. Instantly."
No Flask needed - runs directly inside Streamlit
"""

import re
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

SUPPORTED_LANGUAGES = ["Python", "Java", "C++", "JavaScript", "SQL", "Bash"]


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_code(code: str) -> str:
    code = re.sub(r'#.*',           ' COMMENT ', code)
    code = re.sub(r'//.*',          ' COMMENT ', code)
    code = re.sub(r'/\*.*?\*/',     ' COMMENT ', code, flags=re.DOTALL)
    code = re.sub(r'["\'].*?["\']', ' STRING ',  code)
    code = re.sub(r'\b\d+\.?\d*\b', ' NUM ',     code)
    code = re.sub(r'\s+',           ' ',          code)
    return code.strip().lower()


# ─── Train ────────────────────────────────────────────────────────────────────

def train_models(df: pd.DataFrame, test_size: float = 0.2, max_features: int = 5000) -> dict:
    df = df.dropna(subset=['code', 'language']).copy()
    df['processed'] = df['code'].apply(preprocess_code)

    le = LabelEncoder()
    y  = le.fit_transform(df['language'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['processed'], y,
        test_size=test_size, random_state=42, stratify=y
    )

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    # Naive Bayes
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_tr, y_train)
    nb_preds = nb.predict(X_te)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_tr, y_train)
    lr_preds = lr.predict(X_te)

    metrics = {
        "naive_bayes": {
            "accuracy":         round(accuracy_score(y_test, nb_preds) * 100, 2),
            "confusion_matrix": confusion_matrix(y_test, nb_preds).tolist(),
            "report":           classification_report(
                                    y_test, nb_preds,
                                    target_names=le.classes_,
                                    output_dict=True
                                ),
        },
        "logistic_regression": {
            "accuracy":         round(accuracy_score(y_test, lr_preds) * 100, 2),
            "confusion_matrix": confusion_matrix(y_test, lr_preds).tolist(),
            "report":           classification_report(
                                    y_test, lr_preds,
                                    target_names=le.classes_,
                                    output_dict=True
                                ),
        },
        "languages":      list(le.classes_),
        "total_samples":  len(df),
        "train_size":     len(X_train),
        "test_size":      len(X_test),
    }

    # Persist to disk
    objects = {
        "vectorizer":          vec,
        "naive_bayes":         nb,
        "logistic_regression": lr,
        "label_encoder":       le,
    }
    for name, obj in objects.items():
        with open(f"{MODEL_DIR}/{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
    with open(f"{MODEL_DIR}/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    return {**objects, "metrics": metrics, "trained": True}


# ─── Load ─────────────────────────────────────────────────────────────────────

def load_models() -> dict | None:
    required = ["vectorizer", "naive_bayes", "logistic_regression", "label_encoder"]
    if not all(os.path.exists(f"{MODEL_DIR}/{n}.pkl") for n in required):
        return None
    m = {}
    for name in required:
        with open(f"{MODEL_DIR}/{name}.pkl", "rb") as f:
            m[name] = pickle.load(f)
    if os.path.exists(f"{MODEL_DIR}/metrics.pkl"):
        with open(f"{MODEL_DIR}/metrics.pkl", "rb") as f:
            m["metrics"] = pickle.load(f)
    m["trained"] = True
    return m


# ─── Predict ──────────────────────────────────────────────────────────────────

def predict(models: dict, code: str, model_choice: str = "both") -> dict:
    processed = preprocess_code(code)
    vec_input  = models["vectorizer"].transform([processed])
    le         = models["label_encoder"]

    def _single(clf):
        idx   = clf.predict(vec_input)[0]
        proba = clf.predict_proba(vec_input)[0]
        label = le.inverse_transform([idx])[0]
        conf  = round(float(proba[idx]) * 100, 2)
        probs = {
            le.inverse_transform([i])[0]: round(float(p) * 100, 2)
            for i, p in enumerate(proba)
        }
        return {
            "language":      label,
            "confidence":    conf,
            "probabilities": dict(sorted(probs.items(), key=lambda x: -x[1])),
        }

    result = {}
    if model_choice in ("naive_bayes", "both"):
        result["naive_bayes"] = _single(models["naive_bayes"])
    if model_choice in ("logistic_regression", "both"):
        result["logistic_regression"] = _single(models["logistic_regression"])
    if model_choice == "both":
        nb_c = result["naive_bayes"]["confidence"]
        lr_c = result["logistic_regression"]["confidence"]
        best = result["logistic_regression"] if lr_c >= nb_c else result["naive_bayes"]
        result["ensemble"] = {
            **best,
            "source": "Logistic Regression" if lr_c >= nb_c else "Naive Bayes",
        }
    return result
