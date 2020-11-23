

# load data
from NeuronNetwork import data_loader
from NeuronNetwork.NN import NN

training_data, validation_data, test_data = data_loader.load()
print(test_data)
#print('training_data: {0} / validation_data: {1} / test_data: {2}'.format(len(training_data), len(validation_data), len(test_data)))

# run NN
nn = NN([784, 100, 10])
nn.train(training_data, 30, 100, 3.0)
correct = nn.evaluate(test_data)
test_data = list(test_data)
total = len(test_data)
#print('Evaluation: {0} / {1} = {2}%'.format(correct, total, 100 * correct/total))