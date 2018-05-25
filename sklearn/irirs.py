from sklearn import datasets

iris = datasets.load_iris()

from sklearn.neural_network import MLPClassifier

x, y = iris.data, iris.target

mlp = MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(5,), random_state=1, learning_rate='constant', learning_rate_init=0.01, max_iter=200 ,activation='logistic', momentum=0.9, verbose=True, tol=0.0001)

from sklearn.model_selection import train_test_split

x_treino, x_test, y_treino, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
mlp.fit(x_treino, y_treino)
saidas = mlp.predict(x_test)

print('Saida da rede:\t', saidas)
print('Saida desejada:\t', y_test)

