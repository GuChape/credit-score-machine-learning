import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


tabela = pd.read_csv("clientes.csv")


label_encoders = {}

for coluna in ["profissao", "mix_credito", "comportamento_pagamento"]:
    encoder = LabelEncoder()
    tabela[coluna] = encoder.fit_transform(tabela[coluna])
    label_encoders[coluna] = encoder


X = tabela.drop(columns=["score_credito", "id_cliente"])
y = tabela["score_credito"]


X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3)


modelo = RandomForestClassifier()
modelo.fit(X_treino, y_treino)


previsoes = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes)

print(f"Acurácia do modelo: {acuracia:.2f}")


novos_clientes = pd.read_csv("novos_clientes.csv")

for coluna in ["profissao", "mix_credito", "comportamento_pagamento"]:
    novos_clientes[coluna] = label_encoders[coluna].transform(novos_clientes[coluna])

previsoes_novas = modelo.predict(novos_clientes)

print("\nPrevisão para novos clientes:")
print(previsoes_novas)