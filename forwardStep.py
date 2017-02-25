import numpy as np
import matplotlib.pyplot as plt
class nn1(object):
	
	def __init__(self):
		self.inputLayersSize = 2
		self.outLayersSize = 1
		self.hiddenLayersSize = 3

		self.w1 = np.random.randn(self.inputLayersSize,self.hiddenLayersSize)
		self.w2 = np.random.randn(self.hiddenLayersSize,self.outLayersSize)
	
	def forward(self,x):
		self.z2 = np.dot(x, self.w1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.w2)
		yHat = self.sigmoid(self.z3)
		return yHat

	def sigmoid(self, z):
		return 1/(1 + np.exp(-z))



NN = nn1()
yHat = NN.forward(1)
plt.plot(yHat)
plt.show()
