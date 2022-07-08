import numpy as np
np.random.seed(0)

X = [[1.0, 2.0, 3.0, 2.5],
	 [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class layerDense:
	def __init__(self, nInputs, nNeurons):
		self.weights = 0.1 * np.random.randn(nInputs, nNeurons)
		self.biases = np.zeros((1, nNeurons))
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

layerOne = layerDense(4, 5)
layerTwo = layerDense(5, 2)

layerOne.forward(X)
layerTwo.forward(layerOne.output)
print(layerTwo.output)