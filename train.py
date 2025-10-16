import pandas as pd
import numpy as np
import json, joblib
from pathlib import Path
#from catboost import CatBoostRegressor, Pool
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor

# ---------- CONFIG ----------
DATA_CSV = "emobank.csv"
TEXT_COL = "text"
SPLIT = "split"
V_COL, A_COL, D_COL = "V", "A", "D"
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"

print("Reading data...")
df = pd.read_csv(DATA_CSV)

# Basic sanity
for c in [SPLIT, TEXT_COL, V_COL, A_COL, D_COL]:
    assert c in df.columns, f"Missing column: {c}"

df = df[[SPLIT, TEXT_COL, V_COL, A_COL, D_COL]].dropna().reset_index(drop=True)

print(f"Range of V_COL {df[V_COL].min(), df[V_COL].max()}")
print(f"Range of A_COL {df[A_COL].min(), df[A_COL].max()}")
print(f"Range of D_COL {df[D_COL].min(), df[D_COL].max()}")

train_idx = df[SPLIT].eq("train")
test_idx = df[SPLIT].eq("test")

y_train = df.loc[train_idx, [V_COL, A_COL, D_COL]].to_numpy(dtype=float)
texts_train = df.loc[train_idx, TEXT_COL].astype(str).tolist()

y_test = df.loc[test_idx, [V_COL, A_COL, D_COL]].to_numpy(dtype=float)
texts_test = df.loc[test_idx, TEXT_COL].astype(str).tolist()

encoder = SentenceTransformer(EMB_MODEL)

def embed(sentences: list[str]) -> np.ndarray:
    return np.asarray(encoder.encode(sentences, batch_size=64, normalize_embeddings=True))

print("Embedding texts...")
X_train = embed(texts_train)
X_test = embed(texts_test)

# ---------- multi-target regression ----------
scaler = StandardScaler().fit(y_train)
y_train_scaled = scaler.transform(y_train)
y_test_scaled = scaler.transform(y_test)

SEED = 42

base = RidgeCV(alphas=np.logspace(-3, 3, 13), fit_intercept=True)
model = MultiOutputRegressor(base)
model.fit(X_train, y_train_scaled)

y_pred_z = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_z)

rmse = np.sqrt(((y_test - y_pred) ** 2).mean(axis=0))
r = np.array([pearsonr(y_test[:,i], y_pred[:,i])[0] for i in range(3)])

def concordance_cc(y_true, y_hat):
    mu_t, mu_p = y_true.mean(axis=0), y_hat.mean(axis=0)
    var_t, var_p = y_true.var(axis=0), y_hat.var(axis=0)
    cov = np.mean((y_true-mu_t)*(y_hat-mu_p), axis=0)
    return (2*cov) / (var_t + var_p + (mu_t-mu_p)**2 + 1e-8)

ccc = concordance_cc(y_test, y_pred)

print("RMSE   [V, A, D]:", np.round(rmse, 3))
print("Pearson[V, A, D]:", np.round(r, 3))
print("CCC    [V, A, D]:", np.round(ccc, 3))


# ---------- SAVE MODEL -----------

ARTIFACTS = Path("artifacts_v1")
ARTIFACTS.mkdir(exist_ok=True)

joblib.dump(model, ARTIFACTS / "vad_regressor.pkl")
joblib.dump(scaler, ARTIFACTS / "vad_scaler.pkl")