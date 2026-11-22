import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, average_precision_score

# 1. Przygotowanie danych (analogicznie do skryptu 02)
file_path = "Wednesday-workingHours.pcap_ISCX.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df['is_anomaly'] = np.where(df['Label'] == 'BENIGN', 0, 1)

# Uwaga: Na pełnym pliku Wednesday wszystkie ataki DoS występują w środkowej części dnia.
# Aby w Val i Test znalazły się różne klasy ataków, stosujemy podział warstwowy/blokowy
# lub łączymy wycinki zawierające ataki. Tutaj zastosujemy podział 60/20/20 po przetasowaniu z zachowaniem ziarna
# (w docelowym pipeline'ie połączysz wszystkie dni tygodnia i podzielisz je per scenariusz ataku).
from sklearn.model_selection import train_test_split

X_all = df.drop(columns=['Label', 'is_anomaly'])
y_all = df['is_anomaly']
labels_all = df['Label']

X_train_raw, X_temp, y_train_raw, y_temp, _, labels_temp = train_test_split(
    X_all, y_all, labels_all, test_size=0.4, random_state=42, stratify=labels_all
)
X_val_raw, X_test_raw, y_val, y_test, _, labels_test = train_test_split(
    X_temp, y_temp, labels_temp, test_size=0.5, random_state=42, stratify=labels_temp
)

# Trening WYŁĄCZNIE na ruchu normalnym
X_train_normal = X_train_raw[y_train_raw == 0]
non_zero_var_cols = X_train_normal.columns[X_train_normal.var() > 0]

X_train_normal = X_train_normal[non_zero_var_cols]
X_val_raw = X_val_raw[non_zero_var_cols]
X_test_raw = X_test_raw[non_zero_var_cols]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 2. Eksperyment z 5 powtórzeniami (różne ziarna losowości) + pomiar RAM i czasu
seeds = [42, 100, 2026, 777, 999]
results = []
last_test_preds = None

for seed in seeds:
    tracemalloc.start()
    t0 = time.perf_counter()

    model = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(X_train_scaled)
    train_time = time.perf_counter() - t0

    # Dobór progu na walidacji
    val_scores = -model.decision_function(X_val_scaled)
    prec, rec, thresh = precision_recall_curve(y_val, val_scores)
    f1_val = 2 * (prec * rec) / (prec + rec + 1e-10)
    best_thresh = thresh[np.argmax(f1_val)]

    # Inferencja na teście
    t1 = time.perf_counter()
    test_scores = -model.decision_function(X_test_scaled)
    y_pred = (test_scores >= best_thresh).astype(int)
    infer_time = time.perf_counter() - t1

    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results.append({
        'seed': seed,
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'pr_auc': average_precision_score(y_test, test_scores),
        'train_time_s': train_time,
        'infer_time_s': infer_time,
        'peak_ram_mb': peak_ram / (1024 * 1024)
    })
    last_test_preds = y_pred

res_df = pd.DataFrame(results)
print("=== WYNIKI Z 5 POWTÓRZEŃ (ŚREDNIA ± ODCHYLENIE STANDARDOWE) ===")
for col in ['f1', 'precision', 'recall', 'pr_auc', 'train_time_s', 'infer_time_s', 'peak_ram_mb']:
    print(f"{col:15s}: {res_df[col].mean():.4f} ± {res_df[col].std():.4f}")

# 3. Analiza skuteczności (Recall / Detection Rate) per konkretna klasa ataku (Hipoteza H4)
print("\n=== WYKRYWALNOŚĆ PER KLASA ATAKU (DLA OSTATNIEGO URUCHOMIENIA) ===")
eval_df = pd.DataFrame({
    'Attack_Type': labels_test.values,
    'Actual': y_test.values,
    'Predicted': last_test_preds
})

for attack_name, group in eval_df.groupby('Attack_Type'):
    total = len(group)
    detected = (group['Predicted'] == 1).sum()
    rate = (detected / total) * 100
    status = "FPR (Fałszywe alarmy)" if attack_name == 'BENIGN' else "Detection Rate (Recall)"
    print(f"{attack_name:30s} | Liczba: {total:6d} | Oznaczone jako anomalia: {detected:6d} ({rate:6.2f}% - {status})")