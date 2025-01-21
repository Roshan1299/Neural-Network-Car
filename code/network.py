import random
import copy
from utils import *


class NeuralNetwork:
    def __init__(self, neuron_count):
        self.levels = []
        for i in range(len(neuron_count) - 1):
            self.levels.append(Level(neuron_count[i], neuron_count[i+1]))

    @staticmethod
    def feed_forward(given_inputs, network):
        outputs = Level.feed_forward(given_inputs, network.levels[0])
        for i in range(1, len(network.levels)):
            outputs = Level.feed_forward(outputs, network.levels[i])
        return outputs

    @staticmethod
    def mutate(network, amount=1):
        for level in network.levels:
            for i in range(len(level.biases)):
                level.biases[i] = lerp(
                    level.biases[i], random.uniform(-1, 1), amount)

            for i in range(len(level.weights)):
                for j in range(len(level.weights[i])):
                    level.weights[i][j] = lerp(
                        level.weights[i][j], random.uniform(-1, 1), amount)

    def get_state(self):
        state = {
            "levels": [level.get_state() for level in self.levels]
        }
        return state

    def set_state(self, state):
        self.levels = [Level.from_state(level_state)
                       for level_state in state["levels"]]

    def clone(self):
        clone_network = NeuralNetwork(
            [len(self.levels[0].inputs)] + [len(level.outputs) for level in self.levels])
        clone_network.set_state(self.get_state())
        return clone_network

    def preprocess_inputs(self, inputs):
        max_value = max(inputs)
        if max_value == 0:
            return [0 for i in inputs]
        return [i / max_value for i in inputs]

    def calculate_performance(self, inputs, expected_outputs):
        predictions = self.feed_forward(inputs, self)
        accuracy = sum([1 for pred, exp in zip(
            predictions, expected_outputs) if pred == exp]) / len(expected_outputs)
        return accuracy

    def l2_regularization(self, lambda_value):
        l2_loss = 0
        for level in self.levels:
            for weight_row in level.weights:
                for weight in weight_row:
                    l2_loss += weight ** 2
        return lambda_value * l2_loss


class Level:
    def __init__(self, input_count, output_count, dropout_rate=0.0):
        self.inputs = [0] * input_count
        self.outputs = [0] * output_count
        self.biases = [0] * output_count

        self.weights = [[0 for _ in range(output_count)]
                        for _ in range(input_count)]
        self._randomize()
        self.dropout_rate = dropout_rate

    def _randomize(self):
        for i in range(len(self.inputs)):
            for j in range(len(self.outputs)):
                self.weights[i][j] = random.uniform(-1, 1)
        for i in range(len(self.biases)):
            self.biases[i] = random.uniform(-1, 1)

    @staticmethod
    def feed_forward(given_inputs, level):
        level.inputs = given_inputs
        for i in range(len(level.outputs)):
            sum = 0
            for j in range(len(level.inputs)):
                sum += level.inputs[j]*level.weights[j][i]
            if sum > level.biases[i]:
                level.outputs[i] = 1
            else:
                level.outputs[i] = 0
        level.outputs = [out if random.random(
        ) > level.dropout_rate else 0 for out in level.outputs]
        return level.outputs

    def get_state(self):
        state = {
            "inputs": copy.deepcopy(self.inputs),
            "outputs": copy.deepcopy(self.outputs),
            "biases": copy.deepcopy(self.biases),
            "weights": copy.deepcopy(self.weights)
        }
        return state

    @classmethod
    def from_state(cls, state):
        level = cls(len(state["inputs"]), len(state["outputs"]))
        level.inputs = state["inputs"]
        level.outputs = state["outputs"]
        level.biases = state["biases"]
        level.weights = state["weights"]
        return level
