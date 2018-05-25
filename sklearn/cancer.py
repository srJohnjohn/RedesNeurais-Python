from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

x, y = cancer['data'], cancer['target']

from sklearn.model_selection import train_test_split

x_treino, x_test, y_treino, y_test = train_test_split(x, y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_treino)
x_treino = scaler.transform(x_treino)
x_test = scaler.transform(x_test)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))

mlp.fit(x_treino, y_treino)

prediction = mlp.predict(x_test)

print('Resultado da rede', prediction)
print('Resultado esperado', y_test)
print(mlp.score(x_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))

print(mlp.intercepts_)
print(mlp.coefs_)
