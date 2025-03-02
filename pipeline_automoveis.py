import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Exemplo de dados (substitua por um dataset real)
data = pd.DataFrame({
    'marca': ['Ford', 'Chevrolet', 'Toyota', 'Honda', 'Ford'],
    'ano': [2015, 2016, 2014, 2017, 2013],
    'km_rodados': [50000, 30000, 60000, 20000, 80000],
    'preco': [40000, 45000, 38000, 50000, 35000]
})

# Separando recursos e alvo
X = data.drop(columns=['preco'])
y = data['preco']

# Definição das colunas categóricas e numéricas
categorical_features = ['marca']
numerical_features = ['ano', 'km_rodados']

# Criando os transformadores
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# Criando o pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Separando treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
pipeline.fit(X_train, y_train)

# Fazendo previsões
y_pred = pipeline.predict(X_test)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio: {mse}')
