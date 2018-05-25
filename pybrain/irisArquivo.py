import numpy as np

#lendo arquivos do irirs dataSet
entradas = np.genfromtxt('iris.txt', delimiter=',', usecols=(0,1,2,3))
saidas = np.genfromtxt('iris.txt', delimiter=',', usecols=(4))

print(entradas[:5])
print(saidas[:5])

#separando conjuntos de treinamentos e teste
entradas_treino = np.concatenate((entradas[:35], entradas[50:84], entradas[100:135]))
saidas_treino = np.concatenate((saidas[:35], saidas[50:84], saidas[100:135]))

entradas_teste = np.concatenate((entradas[35:50], entradas[84:100], entradas[135:150]))
saidas_teste = np.concatenate((saidas[35:50], saidas[84:100], saidas[135:150]))

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised import BackpropTrainer

#criado dataset de treinamento de dimenções (4 ,1)
treinamento = SupervisedDataSet(4 ,1)
#adicioanndo entradas e saidas para o dateset de
for i in range(len(entradas_treino)):
    treinamento.addSample(entradas_treino[i], saidas_treino[i])

print(len(treinamento))
print(treinamento.indim)
print(treinamento.outdim)

#criando rede
rede = buildNetwork(treinamento.indim, 2, treinamento.outdim, bias=True)
#definindo algoritmo de treinamento
treiner = BackpropTrainer(rede, treinamento, learningrate=0.01, momentum=0.3)

#treinando
for epoca in range(1000):
    treiner.train()

#criando dataset de treinamento
teste = SupervisedDataSet(4,1)

#testando rede
for i in range(len(entradas_teste)):
    teste.addSample(entradas_teste[i], saidas_teste[i])

treiner.testOnData(teste, verbose=True)

