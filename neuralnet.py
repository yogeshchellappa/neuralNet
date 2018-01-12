import math
import sys
from scipy.io import arff
import numpy as np
import random
import itertools

"""
Preprocess data.
Default numerical encoding for binary classification is 0 for negative samples and 1 for positive samples.
"""		
def preprocessData(data, meta, classificationLabels = [0.0, 1.0]):
	# Create a copy of the data
	labeledData = data[:]

	# Encode the classification labels; the first class label is 0 and the second is 1
	labels = list(meta["Class"][1])

	for i in xrange(len(labeledData)):
		labeledData[i][-1] = float(classificationLabels[0]) if labeledData[i][-1] == labels[0] else float(classificationLabels[1])
	
	return labeledData
	
"""
Generates subsamples
"""
def generateSubSamples(subSamples, subSampleSize):
	# Generate subsamples	
	generatedSubSamples = random.sample(subSamples, subSampleSize)
	
	# Remove the generated samples so that we do not sample them again
	for i in generatedSubSamples:
		subSamples.remove(i)

	return generatedSubSamples
	
"""
Splits the data into 'numberOfFolds' stratified folds
"""
def getStratifiedSamples(labeledData, numberOfFolds, classificationLabels = [0.0, 1.0]):
	sampleSets = []
	
	# First, stratify the data
	negativeSamples = [i for i in labeledData if float(i[-1]) == classificationLabels[0]]
	positiveSamples = [i for i in labeledData if float(i[-1]) == classificationLabels[1]]

	# Next, we compute the number of samples for the given numberOfFolds
	negativeSubSampleSize = len(negativeSamples) / numberOfFolds
	positiveSubSampleSize = len(positiveSamples) / numberOfFolds
	
	# Now we pick positive and negative samples randomly from the two sets
	for i in xrange(numberOfFolds):
		sampleSet = []
		
		# Add negative samples
		sampleSet += generateSubSamples(negativeSamples, negativeSubSampleSize)
		
		# Add positive samples
		sampleSet += generateSubSamples(positiveSamples, positiveSubSampleSize)
		
		# Shuffle the set to distribute the samples
		random.shuffle(sampleSet)
		
		sampleSets.append(sampleSet)
	
	# If we have left over samples, we distribute them across the sampleSets
	combinedSet = negativeSamples + positiveSamples
	random.shuffle(combinedSet)
	
	for i in xrange(len(combinedSet)):
		sampleSets[i % len(sampleSets)].append(combinedSet[i % len(sampleSets)])
	
	return sampleSets

"""
Generates weights randomly in xrange [-1, +1]
"""	
def generateWeights(neuronsPerLayer):
	hiddenWeights = []	
	
	for num in xrange(neuronsPerLayer):
		hiddenWeights.append(np.array(np.random.uniform(-1,1, neuronsPerLayer), dtype = "float64"))
	
	# Hidden bias
	hiddenBiases = np.array(np.random.uniform(-1, 1, neuronsPerLayer), dtype = "float64")
	
	# Generate output layer weights and one bias unit
	outputWeights = np.array(np.random.uniform(-1, 1, neuronsPerLayer), dtype = "float64")
	
	# Finally generate a random bias for output neuron
	outputBias = random.uniform(-1, 1)	
	
	return hiddenWeights, hiddenBiases, outputWeights, outputBias
	
"""
Driver 
"""	
def main():
	trainingFile = sys.argv[1]
	numberOfFolds = int(sys.argv[2])
	learningRate = float(sys.argv[3])
	numberOfEpochs = int(sys.argv[4])
	
	# Read the training data
	data, meta = arff.loadarff(trainingFile)
	
	# Number of neurons
	neuronsPerLayer = len(meta._attrnames[:-1])
	
	# Preprocess data
	labeledData = preprocessData(data, meta)
	
	# Stratified N fold cross validation
	stratifiedSamples = getStratifiedSamples(labeledData, numberOfFolds)
	
	# Keeps track of the values of the prediction results on the test set
	testSetResults = []
	
	for i in xrange(len(stratifiedSamples)):
		# Get the holdout set and the training set
		holdoutSet = stratifiedSamples[i]
		
		trainingSet = stratifiedSamples[:i] + stratifiedSamples[i+1:]
		
		# Flatten list of lists into a single list
		trainingSet = list(itertools.chain.from_iterable(trainingSet))
		trainingSet = [map(float ,record) for record in trainingSet]
		
		hiddenWeights, hiddenBiases, outputWeights, outputBias = generateWeights(neuronsPerLayer)
		
		for epoch in xrange(numberOfEpochs):
			# Training Data
			for record in trainingSet:
				hiddenLayerActivations = []
				delta = []
				
				record = np.array(record, dtype="float")
				actualOutput = record[-1]
				
				# Remove the target label
				record = np.delete(record, -1)
				
				# Input layer to hidden layer
				for hiddenLayerNode in xrange(neuronsPerLayer):
					weights = hiddenWeights[hiddenLayerNode]
					bias = hiddenBiases[hiddenLayerNode]
					net = np.dot(record, weights) + bias
					hiddenLayerActivations.append(1.0 / (1 + math.exp(-net)))
					
				# Hidden Layer to output layer
				out = np.dot(hiddenLayerActivations, outputWeights) + outputBias
				
				# Compute sigmoid
				computedOutput = 1.0 / (1 + math.exp(-out))
				
				# Backpropagation
				backpropConstant = learningRate * (actualOutput - computedOutput)

				# First, update the output weights
				outputWeights = np.add(outputWeights, np.multiply(hiddenLayerActivations, backpropConstant))
				
				# Next, update the output bias
				outputBias += backpropConstant * outputBias			
				
				# Compute the delta at each hidden layer node
				for node in xrange(len(hiddenLayerActivations)):
					delta.append(hiddenLayerActivations[node] * (1 - hiddenLayerActivations[node]) * outputWeights[node] * backpropConstant)
				
				# Update hidden layer biases	
				hiddenBiases = np.add(hiddenBiases, np.multiply(delta, hiddenBiases))
				
				# Update hidden layer weights
				for node in range(neuronsPerLayer):
					hiddenWeights[node] = np.add(hiddenWeights[node], np.multiply(record, delta[node]))
		
		# Test data
		for record in holdoutSet:
			hiddenLayerActivations = []
			
			# Keep track of where this record occurs in the original dataset
			index = np.where(labeledData == record)[0][0]
			
			record = map(float, record)
			actualClass = "Rock" if record[-1] == 0 else "Mine"
	
			# Remove the target label
			record = np.delete(record, -1)
			
			record = np.array(record, dtype="float")
			
			# Input layer to hidden layer
			for hiddenLayerNode in xrange(neuronsPerLayer):
				weights = np.array(hiddenWeights[hiddenLayerNode], dtype='float')
				bias = hiddenBiases[hiddenLayerNode]
				net = np.dot(record, weights) + bias
				hiddenLayerActivations.append(1.0 / (1 + math.exp(-net)))
				
			# Hidden Layer to output layer
			out = np.dot(hiddenLayerActivations, outputWeights) + outputBias	
			
			# Compute sigmoid
			computedOutput = 1.0 / (1 + math.exp(-out))
			
			predictedClass = "Mine" if computedOutput > 0.5 else "Rock"
			
			testSetResults.append([index, i, predictedClass, actualClass, computedOutput])
			
	for i in sorted(testSetResults, key = lambda x : x[0]):
		i = i[1:]		
		print(" ".join(str(x) for x in i))

main()