from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# ======================= ELABORAZIONE DATI TRAINING =============================
STEP = 250
# Lista per memorizzare i segmenti di training
X_train_final = []
y_train_final = []

# Leggi il file CSV
dfSegment = pd.read_csv("data/segments.csv", index_col="timestamp")
channelFix = "CADC0872"

# Itera su ogni segmento unico per il canale corrente
for segment in dfSegment[dfSegment["channel"] == channelFix]["segment"].unique():
    mask = (dfSegment["train"] == 1) & (dfSegment["channel"] == channelFix) & (dfSegment["segment"] == segment)

    # Filtra i dati in base alla maschera
    X_trainS = dfSegment.loc[mask, "value"] #.reset_index(drop=True).values  # Estrarre solo 'value'
    y_trainS = dfSegment.loc[mask, "anomaly"] #.reset_index(drop=True).values  # Estrarre solo 'value'
    # print(X_trainS.shape)
    # Suddividi in sottoliste di STEP elementi
    for i in range(0, len(X_trainS) - STEP + 1, STEP):
        X_train_final.append(X_trainS[i:i + STEP])
        y_train_final.append(y_trainS[i])
        

# Converti la lista in un numpy array
X_train = np.array(X_train_final)
y_train = np.array(y_train_final)
# Reshape per ottenere la shape desiderata
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_train = X_train.transpose(0, 2, 1)
# print(X_train_final.shape)


# ======================= ELABORAZIONE DATI TEST =============================

# Predisposizione del test set
test_data = dfSegment[dfSegment["train"] == 0]
# Predisposizione del test set
X_test_final = []
y_test_final = []

for segment in test_data[test_data["channel"] == channelFix]["segment"].unique():

    mask = (test_data["channel"] == channelFix) & (test_data["segment"] == segment)
    X_testS = test_data.loc[mask, "value"]#.reset_index(drop=True).values
    y_testS = test_data.loc[mask, "anomaly"]#.reset_index(drop=True).values
    
    for i in range(0, len(X_testS) - STEP + 1, STEP):
        X_test_final.append(X_testS[i:i + STEP])
        y_test_final.append(y_testS[i])


X_test = np.array(X_test_final)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_test = X_test.transpose(0, 2, 1)
# print("X_test: ",X_test)
# X_test = np.array(X_test_final).reshape(len(X_test_final), STEP, 1)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


y_test = np.array(y_test_final)
print("y_test: ",y_test)

# ======================= PRE-PROCESSING =============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
