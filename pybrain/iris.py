from sklearn import datasets
from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

iris = datasets.load_iris()

x, y = iris.data, iris.target
print(len(x))

dataset = ClassificationDataSet(4 ,1, nb_classes=3)

for i in range(len(x)):
    dataset.addSample(x[i], y[i])

train_data, part_data = dataset.splitWithProportion(0.6)

test_data, val_data = part_data.splitWithProportion(0.5)

net = buildNetwork(dataset.indim, 3, dataset.outdim)
trainer = BackpropTrainer(net, dataset=train_data, learningrate=0.01, momentum=0.1, verbose=True)

train_errors, val_errors = trainer.trainUntilConvergence(dataset=train_data, maxEpochs=100)

trainer.totalepochs


