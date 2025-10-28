import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score
import pickle

#1
with open(r'risco_credito.pkl', 'rb') as f:
    dados = pickle.load(f)

if isinstance(dados, (list, tuple)) and len(dados) == 2:
    X_arr = np.array(dados[0])
    y_arr = np.array(dados[1]).reshape(-1,1)
    X_arr = X_arr[:, :4]  
    combined = np.hstack([X_arr, y_arr])
    col_names = ['historia','divida','garantias','renda','risco']
    df = pd.DataFrame(combined, columns=col_names)

elif isinstance(dados, np.ndarray):
    if dados.ndim == 1:
        df = pd.DataFrame(dados, columns=['risco'])
    else:
        df = pd.DataFrame(dados[:, :5], columns=['historia','divida','garantias','renda','risco'])
#2
df = df[df['risco'] != 'moderado'] #Remove os registros moderados
#2
encoder = LabelEncoder() #Realizado o enconder
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

X = df[['historia','divida','garantias','renda']].values
y = df['risco'].values
#3
modelo = LogisticRegression(random_state=1)#Treina o algoritmo
modelo.fit(X, y)

#4
print("B0 (intercept_):", modelo.intercept_)#Utiliza o comando intercept_
#5
print("Coeficientes (coef_):", modelo.coef_)#Utiliza o comando coef_


#6 - Testes de previsão
#Codificação manual:
#história: ruim=0, desconhecida=1, boa=2
#dívida: alta=0, baixa=1
#garantias: nenhuma=0, adequada=1
#renda: <15=0, 15–35=1, >35=2

teste_a = np.array([[2,0,0,2]])  # história boa, dívida alta, garantias nenhuma, renda >35
teste_b = np.array([[0,0,1,0]])  # história ruim, dívida alta, garantias adequada, renda <15

prev_a = modelo.predict(teste_a)
prev_b = modelo.predict(teste_b)

encoder_risco = LabelEncoder()
encoder_risco.fit(['baixo','alto'])
prev_a_texto = encoder_risco.inverse_transform(prev_a)
prev_b_texto = encoder_risco.inverse_transform(prev_b)

print("\nPrevisão A (boa, alta, nenhuma, >35):", prev_a_texto[0])
print("Previsão B (ruim, alta, adequada, <15):", prev_b_texto[0])


#7
path_credit = r"credit.pkl"
with open(path_credit, "rb") as f:
    credit = pickle.load(f)
X_train, y_train, X_test, y_test = credit
print(f"Shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

modelo_credit = LogisticRegression(random_state=1, max_iter=1000)#Treina o modelo
modelo_credit.fit(X_train, y_train)
prev_credit = modelo_credit.predict(X_test)#Faz a previsão com base de teste

cm = confusion_matrix(y_test, prev_credit)
acc = accuracy_score(y_test, prev_credit)
prec = precision_score(y_test, prev_credit, average='macro')
rec = recall_score(y_test, prev_credit, average='macro') #Avaliado o desempenho

print("\nMatriz de Confusão:\n", cm)
print(f"\nAcurácia: {acc:.4f}")
print(f"Precisão média: {prec:.4f}")
print(f"Recall médio: {rec:.4f}")
print("\nRelatório completo:\n", classification_report(y_test, prev_credit))

#8
print("""
A Regressão Logística costuma apresentar desempenho intermediário entre naive bayes e random forest. Ela tende a superar o naive bayes quando há correlação entre as variáveis,
porém, modelos como random forest, geralmente tem desempenho superior em dados não lineares.
""")