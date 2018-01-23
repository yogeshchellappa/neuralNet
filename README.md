# Neural Net

Train and test a neural network with one hidden layer using backpropagation. Specifically:
* This is intended for binary classification problems.
* All of the attributes are numeric.
* The neural network has connections between input and the hidden layer, and between the hidden and output layer and one bias unit and one output node.
* The number of units in the hidden layer should be equal to the number of input units.
* For training the neural network, use n-fold stratified cross validation.
* Use sigmoid activation function and train using stochastic gradient descent.
* Randomly set initial weights for all units including bias in the range (-1,1).
* Use a threshold value of 0.5. If the sigmoidal output is less than 0.5, take the prediction to be the class listed first in the ARFF file in the class attributes section; else take the prediction to be the class listed second in the ARFF file.

## Usage Instructions
The program should be callable from command line as follows: </br>
<code>neuralnet.py trainfile numOfFolds learningRate numEpochs</code>
