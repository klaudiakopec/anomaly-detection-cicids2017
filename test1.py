import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score

# 1. Wczytanie i czyszczenie danych
file_path = "Wednesday-workingHours.pcap_ISCX.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df['is_anomaly'] = np.where(df['Label'] == 'BENIGN', 0, 1)

# 2. Podział chronologiczny bez wycieku: 60% Train, 20% Validation, 20% Test
n = len(df)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

X_all = df.drop(columns=['Label', 'is_anomaly'])
y_all = df['is_anomaly']

X_train_raw, y_train_raw = X_all.iloc[:train_end], y_all.iloc[:train_end]
X_val_raw, y_val = X_all.iloc[train_end:val_end], y_all.iloc[train_end:val_end]
X_test_raw, y_test = X_all.iloc[val_end:], y_all.iloc[val_end:]

# Trening WYŁĄCZNIE na ruchu normalnym (klasa 0)
X_train_normal = X_train_raw[y_train_raw == 0]

# Usunięcie kolumn o zerowej wariancji (wykrytych wyłącznie na zbiorze treningowym)
non_zero_var_cols = X_train_normal.columns[X_train_normal.var() > 0]
X_train_normal = X_train_normal[non_zero_var_cols]
X_val_raw = X_val_raw[non_zero_var_cols]
X_test_raw = X_test_raw[non_zero_var_cols]

# Standaryzacja dopasowana TYLKO na zbiorze treningowym
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 3. Trening Isolation Forest
model_if = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
model_if.fit(X_train_scaled)

# 4. Wybór optymalnego progu WYŁĄCZNIE na zbiorze walidacyjnym
val_scores = -model_if.decision_function(X_val_scaled)
precisions, recalls, thresholds = precision_recall_curve(y_val, val_scores)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Optymalny próg wyznaczony na walidacji: {best_threshold:.4f} (Val F1: {f1_scores[best_idx]:.4f})")

# 5. Ostateczna ewaluacja na zamrożonym zbiorze testowym
test_scores = -model_if.decision_function(X_test_scaled)
y_test_pred = (test_scores >= best_threshold).astype(int)

print(f"Test PR-AUC: {average_precision_score(y_test, test_scores):.4f}")
print("\n--- Finalny wynik na zbiorze testowym ---")
print(classification_report(y_test, y_test_pred, target_names=['Norma (0)', 'Anomalia (1)']))