import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, average_precision_score

#AUDYT DANYCH

# Ustawienie ziarna losowości (wymóg reprodukowalności eksperymentu)
RANDOM_STATE = 42

# Wczytanie pliku (upewnij się, że nazwa i ścieżka zgadzają się z Twoim folderem data/)
file_path = "Wednesday-workingHours.pcap_ISCX.csv"
df = pd.read_csv(file_path)

# Usunięcie zbędnych spacji z nazw kolumn (w CIC-IDS2017 nazwy często mają spacje na początku, np. ' Destination Port')
df.columns = df.columns.str.strip()

print(f"Wymiary zbioru: {df.shape[0]} wierszy, {df.shape[1]} kolumn")

#PREPROCESSING

# 1. Sprawdzenie rozkładu klas (ruch normalny vs rodzaje ataków)
print("--- Rozkład etykiet przed czyszczeniem ---")
print(df['Label'].value_counts())

# 2. Zamiana wartości nieskończonych (Inf) na NaN i usunięcie braków
df.replace([np.inf, -np.inf], np.nan, inplace=True)
missing_before = df.isna().sum().sum()
df.dropna(inplace=True)
print(f"\nUsunięto {missing_before} brakujących/nieskończonych wartości. Nowy rozmiar: {df.shape[0]} wierszy.")

# 3. Utworzenie etykiety binarnej: 0 = ruch normalny (BENIGN), 1 = anomalia (Atak)
df['is_anomaly'] = np.where(df['Label'] == 'BENIGN', 0, 1)
print("\n--- Rozkład binarny (0 = Norma, 1 = Anomalia) ---")
print(df['is_anomaly'].value_counts())

# Na potrzeby szybkiego mini-eksperymentu bierzemy próbkę (np. 100 000 wierszy z zachowaniem kolejności)
df_sample = df.iloc[:100000].copy()

# Oddzielamy cechy numeryczne (X) od etykiet (y)
# Usuwamy kolumny tekstowe/etykiety
X = df_sample.drop(columns=['Label', 'is_anomaly'])
y = df_sample['is_anomaly']

# Dzielimy chronologicznie (70% część treningowa, 30% część testowa)
split_idx = int(len(df_sample) * 0.7)
X_train_raw, X_test_raw = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_raw, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# KLUCZOWE: Algorytmy nienadzorowane uczymy WYŁĄCZNIE na ruchu normalnym (y == 0)
X_train_normal = X_train_raw[y_train_raw == 0]

# Dopasowujemy skaler TYLKO na zbiorze treningowym (brak wycieku danych ze zbioru testowego!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_scaled = scaler.transform(X_test_raw)

print(f"Zbiór treningowy (tylko norma): {X_train_scaled.shape[0]} rekordów")
print(f"Zbiór testowy (norma + ataki): {X_test_scaled.shape[0]} rekordów (w tym anomalii: {y_test.sum()})")


# Inicjalizacja modelu Isolation Forest
model_if = IsolationForest(
    n_estimators=100,
    contamination=0.1, # wstępny próg zanieczyszczenia (w docelowej pracy dobierzesz go na zbiorze walidacyjnym)
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# 1. Pomiar czasu uczenia (Training Time)
start_train = time.perf_counter()
model_if.fit(X_train_scaled)
train_time = time.perf_counter() - start_train

# 2. Pomiar czasu inferencji (Inference Time)
start_infer = time.perf_counter()
# Isolation Forest zwraca: 1 dla normy, -1 dla anomalii -> mapujemy na 0 i 1
raw_preds = model_if.predict(X_test_scaled)
y_pred = np.where(raw_preds == -1, 1, 0)
infer_time = time.perf_counter() - start_infer

# Pobranie ciągłego wyniku anomalii (potrzebne do metryki PR-AUC niezależnej od progu)
anomaly_scores = -model_if.decision_function(X_test_scaled)

# 3. Wyniki i metryki
print(f"Czas uczenia: {train_time:.4f} s")
print(f"Czas inferencji ({len(X_test_scaled)} próbek): {infer_time:.4f} s")
print(f"Przepustowość (Throughput): {len(X_test_scaled) / infer_time:.0f} przepływów/sek.")
print(f"PR-AUC Score: {average_precision_score(y_test, anomaly_scores):.4f}\n")

print("--- Raport klasyfikacji ---")
print(classification_report(y_test, y_pred, target_names=['Norma (0)', 'Anomalia (1)']))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Norma (0)', 'Anomalia (1)'])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title("Macierz pomyłek – Isolation Forest (CIC-IDS2017)")
plt.tight_layout()
plt.show()