#Finished 11 May 2021
#Scalable point classifier, made from FFN_General

import numpy as np
import math

class MLN:
    def __init__(self, learningRate, hiddenLayerCount, neuronsPerHiddenLayer, breakPoint, trainingInputs, desiredTrainingOutputs, testingInputs, desiredTestingOutputs):
        print('Program Started\n')

        def hiddenActivation(num): return 0.01 * num if num <= 0 else num
        def hiddenActivationDerivative(num): return 0.01 if num <= 0 else 1

        def outputActivation(num):
            if num <= 709 and num >= -709: return 1 / (1 + np.exp(-1 * num))
            elif num > 709: return 1.0
            else: return 0.0
        def outputActivationDerivative(num): return outputActivation(num) * (1 - outputActivation(num))

        def feedForward(input, desiredOutput):
            actualHiddenLayers = np.array([[0.0] * (neuronsPerHiddenLayer + 1)] * hiddenLayerCount)
            activatedHiddenLayers = np.array([[0.0] * (neuronsPerHiddenLayer + 1)] * hiddenLayerCount)
            actualOutputLayer = np.array(([0.0] * self.numOutput))
            activatedOutputLayer = np.array(([0.0] * self.numOutput))
            cost = 0

            for i in range(hiddenLayerCount):
                for j in range(neuronsPerHiddenLayer):
                    if i != 0:
                        actualHiddenLayers[i][j] = np.dot(activatedHiddenLayers[i - 1], self.hiddenWeights[i - 1].T[j])
                        activatedHiddenLayers[i][j] = hiddenActivation(actualHiddenLayers[i][j])
                    else:
                        actualHiddenLayers[0][j] = np.dot(input, self.inputWeights.T[j])
                        activatedHiddenLayers[0][j] = hiddenActivation(actualHiddenLayers[0][j])

                    actualHiddenLayers[i][neuronsPerHiddenLayer] = 1.0
                    activatedHiddenLayers[i][neuronsPerHiddenLayer] = 1.0

            for i in range(self.numOutput):
                actualOutputLayer[i] = np.dot(activatedHiddenLayers[hiddenLayerCount - 1], self.outputWeights.T[i])
                activatedOutputLayer[i] = outputActivation(actualOutputLayer[i])
                cost += math.pow((activatedOutputLayer[i] - desiredOutput[i]), 2)
            cost *= (1 / self.numOutput)

            self.actualHiddenLayers = actualHiddenLayers
            self.activatedHiddenLayers = activatedHiddenLayers
            self.actualOutputLayer = actualOutputLayer
            self.activatedOutputLayer = activatedOutputLayer
            self.desiredOutputArray = desiredOutput
            self.cost = cost

        def backpropagate(inputs, desiredOutputs):
            outputWeightAdjustments = np.zeros((neuronsPerHiddenLayer + 1, self.numOutput))
            inputWeightAdjustments = np.zeros((self.numInput, neuronsPerHiddenLayer))
            if hiddenLayerCount > 1: hiddenWeightsAdjustments = np.zeros((hiddenLayerCount - 1, neuronsPerHiddenLayer + 1, neuronsPerHiddenLayer))
            else: hiddenWeightsAdjustments = None

            for h in range(len(inputs)):
                if h % (len(inputs) / 50) == 0: print('Training: ' + str(round(h / len(inputs) * 100, 2)) + '%', end='\r')

                costDerivativeRespectToHiddenNeurons = np.array([[0.0] * (neuronsPerHiddenLayer + 1)] * hiddenLayerCount)
                costDerivativeRespectToOutputNeurons = np.array(([0.0] * self.numOutput))

                feedForward(inputs[h], desiredOutputs[h])

                for i in range(len(costDerivativeRespectToOutputNeurons)):
                    costDerivativeRespectToOutputNeurons[i] = (2 * (1 / self.numOutput)) * (self.activatedOutputLayer[i] - self.desiredOutputArray[i]) * outputActivationDerivative(self.actualOutputLayer[i])

                for i in range(len(costDerivativeRespectToHiddenNeurons))[::-1]:
                    for j in range(len(costDerivativeRespectToHiddenNeurons[i])):
                        if i != (hiddenLayerCount - 1):
                            for k in range(neuronsPerHiddenLayer):
                                costDerivativeRespectToHiddenNeurons[i][j] += (costDerivativeRespectToHiddenNeurons[i + 1][k] * self.hiddenWeights[i][j][k])
                        else:
                            for l in range(self.numOutput):
                                costDerivativeRespectToHiddenNeurons[i][j] += costDerivativeRespectToOutputNeurons[l] * self.outputWeights[j][l]
                        costDerivativeRespectToHiddenNeurons[i][j] *= hiddenActivationDerivative(self.actualHiddenLayers[i][j])

                for i in range(len(outputWeightAdjustments)):
                    for j in range(len(outputWeightAdjustments[i])):
                        outputWeightAdjustments[i][j] = costDerivativeRespectToOutputNeurons[j] * self.activatedHiddenLayers[hiddenLayerCount - 1][i]

                try:
                    for i in range(len(hiddenWeightsAdjustments)):
                        for j in range(len(hiddenWeightsAdjustments[i])):
                            for k in range(len(hiddenWeightsAdjustments[i][j])):
                                hiddenWeightsAdjustments[i][j][k] = costDerivativeRespectToHiddenNeurons[i + 1][k] * self.activatedHiddenLayers[i][j]
                except: pass

                for i in range(len(inputWeightAdjustments)):
                    for j in range(len(inputWeightAdjustments[i])):
                        inputWeightAdjustments[i][j] = costDerivativeRespectToHiddenNeurons[0][j] * inputs[h][i]

                self.inputWeights += -1 * inputWeightAdjustments * self.learningRate
                try: self.hiddenWeights += -1 * hiddenWeightsAdjustments * self.learningRate
                except: pass
                self.outputWeights += -1 * outputWeightAdjustments * self.learningRate

        def test(inputs, desiredOutputs):
            totalTested = 0
            correctTested = 0

            for h in range(len(inputs)):
                feedForward(inputs[h], desiredOutputs[h])
                if self.cost <= 0.01: correctTested += 1
                totalTested += 1

            self.testAccuracy = round(correctTested / totalTested * 100, 2)
            print('Test Accuracy: ' + str(self.testAccuracy) + '%')

        self.learningRate = learningRate

        self.trainingInputs = trainingInputs
        self.testingInputs = testingInputs

        self.desiredTrainingOutputs = desiredTrainingOutputs
        self.desiredTestingOutputs = desiredTestingOutputs

        self.numInput = len(self.trainingInputs[0])
        self.numOutput = len(self.desiredTrainingOutputs[0])

        self.inputWeights = np.random.standard_normal((self.numInput, neuronsPerHiddenLayer))
        if hiddenLayerCount > 1: self.hiddenWeights = np.random.standard_normal((hiddenLayerCount - 1, neuronsPerHiddenLayer + 1, neuronsPerHiddenLayer))
        else: self.hiddenWeights = None
        self.outputWeights = np.random.standard_normal((neuronsPerHiddenLayer + 1, self.numOutput))

        self.testAccuracy = 0.0
        self.iterationCount = 0

        while True:
            self.iterationCount += 1
            print('Iteration:', self.iterationCount)

            trainShuffler = np.random.permutation(len(self.trainingInputs))
            testShuffler = np.random.permutation(len(self.testingInputs))

            self.trainingInputs = self.trainingInputs[trainShuffler]
            self.desiredTrainingOutputs = self.desiredTrainingOutputs[trainShuffler]

            backpropagate(self.trainingInputs, self.desiredTrainingOutputs)

            self.testingInputs = self.testingInputs[testShuffler]
            self.desiredTestingOutputs = self.desiredTestingOutputs[testShuffler]

            test(self.testingInputs, self.desiredTestingOutputs)

            print('\n')

            if self.testAccuracy >= breakPoint: break

        print('Final Test')
        test(self.testingInputs, self.desiredTestingOutputs)

def lineFunction(x):
    #return 1 / 4 * (x * x) - 3
    return 8 * math.sin(0.4 * x)

def createInputs(pointDistanceFromZero, pointCount):
    inputs = (2 * pointDistanceFromZero) * np.random.random((pointCount, 3)) - pointDistanceFromZero
    for i in range(len(inputs)):
        inputs[i][0] = round(inputs[i][0], 2)
        inputs[i][1] = round(inputs[i][1], 2)
        inputs[i][2] = 1
    return inputs

def createOutputs(inputs):
    outputs = np.array([[0]] * len(inputs))
    for i in range(len(inputs)): outputs[i][0] = 1 if lineFunction(inputs[i][0]) < inputs[i][1] else 0
    return outputs

if __name__=='__main__':
    learningRate = 0.01
    hiddenLayerCount = 3
    neuronsPerHiddenLayer = 10
    breakPoint = 93

    trainingInputs = createInputs(10, 50000)
    desiredTrainingOutputs = createOutputs(trainingInputs)
    testingInputs = createInputs(10, 10000)
    desiredTestingOutputs = createOutputs(testingInputs)

    network = MLN(learningRate, hiddenLayerCount, neuronsPerHiddenLayer, breakPoint, trainingInputs, desiredTrainingOutputs, testingInputs, desiredTestingOutputs)