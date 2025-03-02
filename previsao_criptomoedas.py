import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Baixando dados históricos do Bitcoin usando CoinGecko API
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30&interval=daily"
response = requests.get(url)
data = response.json()

# Transformando em DataFrame
df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Criando variável alvo (1 = preço sobe no dia seguinte, 0 = preço cai)
df['target'] = (df['price'].shift(-1) > df['price']).astype(int)

# Selecionando recursos e removendo valores nulos
df = df.dropna()
X = df[['price']]
y = df['target']

# Separando treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazendo previsões e avaliando
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')
