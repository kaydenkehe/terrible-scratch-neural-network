#Finished March 2021

#Classify points as being below or above a quadratic line
#Structured like this (Multilayer Nueron Network / Feedforward Network):

#This network uses the stochastic gradient descent method (I think?????), it just calculates the gradient for every single point idrk

#Layer 1 (Input): 3 neurons, one for x coordinate of point, one for y coordinate of point, one for bias
#Layer 2 (Hidden): 3 neurons, two are the sum of activations and weights of previous layer and one is for bias
#Layer 3 (Output): 1 neuron, tries to return 1 if point is above line and 0 if below

import numpy as np
import math
import time

#def sigmoid(num): return 1 / (1 + np.exp(-1 * num))
def relu(num): return 0 if num <= 0 else num

def createInputs(pointCount, pointDistanceFromZero):
    inputs = (2 * pointDistanceFromZero) * np.random.random((pointCount, 3)) - pointDistanceFromZero
    for i in range(len(inputs)):
        inputs[i][0] = round(inputs[i][0], 2)
        inputs[i][1] = round(inputs[i][1], 2)
        inputs[i][2] = 1
    return inputs

def createOutputs(inputs, a, b, c):
    outputs = np.array([[0]] * len(inputs))
    for i in range(len(inputs)):
        outputs[i][0] = 1 if (a * math.pow(inputs[i][0], 2)) + (b * inputs[i][0]) + c < inputs[i][1] else 0
    return outputs

#This function finds the derivative of the cost with respect to every single weight individually
def costDerivativeRespectToWeights(actualLayer2Neuron1, actualLayer2Neuron2, reluLayer2Neuron1, reluLayer2Neuron2, reluOutput, actualOutput, desiredOutputs, inputs, i, layer):
    costDerivativeRespectToReluOutput = reluOutput - desiredOutputs[i]
    #sigmoidDerivativeRespectToActualOutput = reluOutput * (1 - reluOutput)
    reluDerivativeRespectToActualOutput = 0 if reluOutput <= 0 else 1

    actualOutputDerivativeRespectToReluLayer2Neuron1 = weightsL2[0][0]
    actualOutputDerivativeRespectToReluLayer2Neuron2 = weightsL2[0][1]

    #sigmoidLayer2Neuron1DerivativeRespectToActualLayer2Neuron1 = reluLayer2Neuron1 * (1 - reluLayer2Neuron1)
    reluLayer2Neuron1DerivativeRespectToActualLayer2Neuron1 = 0 if reluLayer2Neuron1 <= 0 else 1
    #sigmoidLayer2Neuron2DerivativeRespectToActualLayer2Neuron2 = reluLayer2Neuron2 * (1 - reluLayer2Neuron2)
    reluLayer2Neuron2DerivativeRespectToActualLayer2Neuron2 = 0 if reluLayer2Neuron2 <= 0 else 1

    costDerivativeRespectToActualOutput = costDerivativeRespectToReluOutput * reluDerivativeRespectToActualOutput
    actualOutputDerivativeRespectToActualLayer2Neuron1 = actualOutputDerivativeRespectToReluLayer2Neuron1 * reluLayer2Neuron1DerivativeRespectToActualLayer2Neuron1
    actualOutputDerivativeRespectToActualLayer2Neuron2 = actualOutputDerivativeRespectToReluLayer2Neuron2 * reluLayer2Neuron2DerivativeRespectToActualLayer2Neuron2
    costDerivativeRespectToActualLayer2Neuron1 = costDerivativeRespectToActualOutput * actualOutputDerivativeRespectToActualLayer2Neuron1
    costDerivativeRespectToActualLayer2Neuron2 = costDerivativeRespectToActualOutput * actualOutputDerivativeRespectToActualLayer2Neuron2


    actualOutputDerivativeRespectToReluLayer2Neuron1Weight = reluLayer2Neuron1
    actualOutputDerivativeRespectToReluLayer2Neuron2Weight = reluLayer2Neuron2
    actualOutputDerivativeRespectToLayer2Bias = 1

    actualLayer2Neuron1DerivativeRespectToLayer1Neuron1Weight1 = inputs[i][0]
    actualLayer2Neuron1DerivativeRespectToLayer1Neuron2Weight1 = inputs[i][1]
    actualLayer2Neuron1DerivativeRespectToLayer1Bias1 = 1

    actualLayer2Neuron2DerivativeRespectToLayer1Neuron1Weight2 = inputs[i][0]
    actualLayer2Neuron2DerivativeRespectToLayer1Neuron2Weight2 = inputs[i][1]
    actualLayer2Neuron2DerivativeRespectToLayer1Bias2 = 1

    if layer == 2:
        return np.array([[
            costDerivativeRespectToActualOutput * actualOutputDerivativeRespectToReluLayer2Neuron1Weight,
            costDerivativeRespectToActualOutput * actualOutputDerivativeRespectToReluLayer2Neuron2Weight,
            costDerivativeRespectToActualOutput * actualOutputDerivativeRespectToLayer2Bias
        ]]).reshape(1,3)
    elif layer == 1:
        return np.array(([
            costDerivativeRespectToActualLayer2Neuron1 * actualLayer2Neuron1DerivativeRespectToLayer1Neuron1Weight1,
            costDerivativeRespectToActualLayer2Neuron1 * actualLayer2Neuron1DerivativeRespectToLayer1Neuron2Weight1,
            costDerivativeRespectToActualLayer2Neuron1 * actualLayer2Neuron1DerivativeRespectToLayer1Bias1
        ], [
            costDerivativeRespectToActualLayer2Neuron2 * actualLayer2Neuron2DerivativeRespectToLayer1Neuron1Weight2,
            costDerivativeRespectToActualLayer2Neuron2 * actualLayer2Neuron2DerivativeRespectToLayer1Neuron2Weight2,
            costDerivativeRespectToActualLayer2Neuron2 * actualLayer2Neuron2DerivativeRespectToLayer1Bias2
        ])).reshape(2,3)

    #Each neuron on layer one has two weights, one that affects neuron one of layer two and one that affects neuron two of layer two.


if __name__=='__main__':
    startTime = time.time()
    learningRate = 0.005
    trainingPointCount = 1000000
    testingPointCount = 5000
    pointDistanceFromZero = 10
    #These are used as ax^2+bx+c to form the line the points will be classified using.
    a = 0.1; b = 1.5; c = -3

    trainingInputs = createInputs(trainingPointCount, pointDistanceFromZero)
    desiredTrainingOutputs = createOutputs(trainingInputs, a, b, c)
    testingInputs = createInputs(testingPointCount, pointDistanceFromZero)
    desiredTestingOutputs = createOutputs(testingInputs, a, b, c)

    #These establish weights for layers 1 and 2.
    weightsL1 = 2 * np.random.random((2, 3)) - 1
    weightsL2 = 2 * np.random.random((1, 3)) - 1

    for i in range(len(trainingInputs)):
        #This network has 1 hidden layer with 2 neurons.
        actualLayer2Neuron1 = np.dot(trainingInputs[i], weightsL1[0])
        actualLayer2Neuron2 = np.dot(trainingInputs[i], weightsL1[1])

        reluLayer2Neuron1 = relu(actualLayer2Neuron1)
        reluLayer2Neuron2 = relu(actualLayer2Neuron2)

        #There's a neuron with value 1 at the end of the layer to introduce a bias, which adds some complexity.
        layer2 = np.array([float(reluLayer2Neuron1), float(reluLayer2Neuron2), 1.0])

        actualOutput = np.dot(layer2, weightsL2[0])
        reluOutput = relu(actualOutput)
        cost = math.pow((reluOutput - desiredTrainingOutputs[i]), 2)

        adjustmentsL1 = costDerivativeRespectToWeights(actualLayer2Neuron1, actualLayer2Neuron2, reluLayer2Neuron1,
                                                       reluLayer2Neuron2, reluOutput, actualOutput,
                                                       desiredTrainingOutputs, trainingInputs, i, 1)
        adjustmentsL2 = costDerivativeRespectToWeights(actualLayer2Neuron1, actualLayer2Neuron2, reluLayer2Neuron1,
                                                       reluLayer2Neuron2, reluOutput, actualOutput,
                                                       desiredTrainingOutputs, trainingInputs, i, 2)
        adjustmentsL1 = -1 * adjustmentsL1 * learningRate
        adjustmentsL2 = -1 * adjustmentsL2 * learningRate
        weightsL1 += adjustmentsL1
        weightsL2 += adjustmentsL2

        if i % 1000000 == 0: print(str(round(i / trainingPointCount * 100, 2)) + '% Training Completed')
    print('100.0% Training Completed')

    #This part tests the weights and gives their accuracy
    correctTested = 0
    totalTested = 0
    for i in range(len(testingInputs)):
        actualLayer2Neuron1 = np.dot(testingInputs[i], weightsL1[0])
        actualLayer2Neuron2 = np.dot(testingInputs[i], weightsL1[1])

        reluLayer2Neuron1 = relu(actualLayer2Neuron1)
        reluLayer2Neuron2 = relu(actualLayer2Neuron2)

        layer2 = np.array([float(reluLayer2Neuron1), float(reluLayer2Neuron2), 1.0])

        actualOutput = np.dot(layer2, weightsL2[0])
        reluOutput = relu(actualOutput)
        cost = math.pow((reluOutput - desiredTestingOutputs[i]), 2) / 2

        if round(cost, 4) <= 0.01: correctTested += 1
        totalTested += 1

    #Basic output stuff, the first three prints are for me to enter in to Desmos and test the weights against an actual graph
    print('\nWeights for layer 1:\n' + str(weightsL1))
    print('Weights for layer 2:\n' + str(weightsL2))
    print('\nFor Desmos:')
    print('c = ' + str(float(weightsL1[0][0])) + 'a + ' +  str(float(weightsL1[0][1])) + 'b + ' + str(float(weightsL1[0][2])))
    print('d = ' + str(float(weightsL1[1][0])) + 'a + ' + str(float(weightsL1[1][1])) + 'b + ' + str(float(weightsL1[1][2])))
    print('h = ' + str(float(weightsL2[0][0])) + 'f + ' + str(float(weightsL2[0][1])) + 'g + ' + str(float(weightsL2[0][2])))
    print('\nTest Accuracy: ' + str(round(correctTested / totalTested * 100, 4)) + '%')
    endTime = time.time()
    print(f'Time it took: {round(endTime - startTime, 2)}')