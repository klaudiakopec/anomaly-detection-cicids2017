import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)

# 1. Wczytanie i czyszczenie danych
file_path = ("Wednesday-workingHours.pcap_ISCX.csv")
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df['is_anomaly'] = np.where(df['Label'] == 'BENIGN', 0, 1)

# 2. PODZIAŁ SCENARIUSZOWY (Chronologiczny wewnątrz każdej klasy, bez losowego mieszania)
# - Ruch BENIGN: 60% na trening (Train), 20% na walidację (Val), 20% na test (Test)
# - Każdy typ ataku osobno: 50% (pierwsza połowa incydentu) -> Val, 50% (druga połowa) -> Test
train_parts, val_parts, test_parts = [], [], []

for label, group in df.groupby('Label', sort=False):
    n = len(group)
    if label == 'BENIGN':
        t_end = int(n * 0.6)
        v_end = int(n * 0.8)
        train_parts.append(group.iloc[:t_end])
        val_parts.append(group.iloc[t_end:v_end])
        test_parts.append(group.iloc[v_end:])
    else:
        mid = int(n * 0.5)
        val_parts.append(group.iloc[:mid])
        test_parts.append(group.iloc[mid:])

df_train = pd.concat(train_parts)
df_val = pd.concat(val_parts)
df_test = pd.concat(test_parts)

# Oddzielenie cech i etykiet
feature_cols = [c for c in df.columns if c not in ['Label', 'is_anomaly']]

X_train = df_train[feature_cols]
X_val, y_val = df_val[feature_cols], df_val['is_anomaly'].values
X_test, y_test = df_test[feature_cols], df_test['is_anomaly'].values
labels_test = df_test['Label'].values

# Usunięcie cech o zerowej wariancji (wyłącznie na podstawie Train)
non_zero_cols = X_train.columns[X_train.var() > 0]
X_train = X_train[non_zero_cols]
X_val = X_val[non_zero_cols]
X_test = X_test[non_zero_cols]

# Skalowanie dopasowane wyłącznie na Train (brak data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Train (wyłącznie BENIGN): {len(X_train_scaled)} rekordów")
print(f"Validation: {len(X_val_scaled)} rekordów (w tym ataków: {y_val.sum()})")
print(f"Test:       {len(X_test_scaled)} rekordów (w tym ataków: {y_test.sum()})\n")

# 3. Trening modelu z pomiarem czasu i pamięci RAM
tracemalloc.start()
t0 = time.perf_counter()

model = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled)

train_time = time.perf_counter() - t0
_, peak_ram_train = tracemalloc.get_traced_memory()
tracemalloc.stop()

# 4. Wybór progu na Validation przy ustalonym limicie fałszywych alarmów (RQ1: np. MAX_FPR = 2%)
MAX_FPR = 0.02
val_scores = -model.decision_function(X_val_scaled)

# Wyznaczamy próg jako (1 - MAX_FPR) percentyl wyników na czystym ruchu w zbiorze walidacyjnym
val_benign_scores = val_scores[y_val == 0]
threshold_fpr = np.quantile(val_benign_scores, 1.0 - MAX_FPR)

# 5. Finalna ewaluacja na Test (wymagane metryki z pkt 5.12 promotora)
t1 = time.perf_counter()
test_scores = -model.decision_function(X_test_scaled)
y_pred = (test_scores >= threshold_fpr).astype(int)
infer_time = time.perf_counter() - t1

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr_test = fp / (fp + tn)
fp_per_10k = (fp / len(y_test)) * 10000
throughput = len(X_test_scaled) / infer_time
latency_ms = (infer_time / len(X_test_scaled)) * 1000

print("=== METRYKI ZBIORCZE NA ZBIORZE TESTOWYM (BUDŻET FPR <= 2%) ===")
print(f"Próg decyzyjny (z Val):     {threshold_fpr:.4f}")
print(f"Precision (Anomalia):       {precision_score(y_test, y_pred):.4f}")
print(f"Recall (Anomalia):          {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score (Anomalia):        {f1_score(y_test, y_pred):.4f}")
print(f"PR-AUC:                     {average_precision_score(y_test, test_scores):.4f}")
print(f"FPR (False Positive Rate):  {fpr_test:.4%} (limit: {MAX_FPR:.2%})")
print(f"Fałszywe alarmy / 10k szt.: {fp_per_10k:.1f}")
print(f"Czas uczenia:               {train_time:.4f} s")
print(f"Przepustowość (Throughput): {throughput:.0f} przepływów/s")
print(f"Śr. opóźnienie (Latency):   {latency_ms:.5f} ms/przepływ")
print(f"Szczytowy RAM (uczenie):    {peak_ram_train / (1024**2):.2f} MB\n")

print("=== WYNIKI PER KLASA ATAKU (HIPOTEZA H4) ===")
eval_df = pd.DataFrame({'Attack_Type': labels_test, 'Predicted': y_pred})
for attack_name, grp in eval_df.groupby('Attack_Type'):
    tot = len(grp)
    det = (grp['Predicted'] == 1).sum()
    pct = (det / tot) * 100
    role = "FPR (fałszywe alarmy)" if attack_name == 'BENIGN' else "Detection Rate (Recall)"
    print(f"{attack_name:28s} | Próbki: {tot:6d} | Alarmy: {det:6d} ({pct:6.2f}% - {role})")