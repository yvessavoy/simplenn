import math

from matrix import Matrix


class ActivationFunction:
    def __init__(self, func, dfunc):
        self.func = func
        self.dfunc = dfunc

sigmoid = ActivationFunction(
    lambda x: 1 / (1 + math.exp(-x)),
    lambda y: y * (1 - y)
)

tanh = ActivationFunction(
    lambda x: math.tanh(x),
    lambda y: 1 - (y * y)
)

class NeuralNetwork:
    def __init__(self, in_nodes, hid_nodes, out_nodes):
        if type(in_nodes) == NeuralNetwork:
            a = in_nodes
            self.input_nodes = a.input_nodes
            self.hidden_nodes = a.hidden_nodes
            self.output_nodes = a.output_nodes

            self.weights_ih = a.weights_ih.copy()
            self.weights_ho = a.weights_ho.copy()

            self.bias_h = a.bias_h.copy()
            self.biah_o = a.bias_o.copy()
        else:
            self.input_nodes = in_nodes
            self.hidden_nodes = hid_nodes
            self.output_nodes = out_nodes

            self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
            self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
            self.weights_ih.randomize()
            self.weights_ho.randomize()

            self.bias_h = Matrix(self.hidden_nodes, 1)
            self.bias_o = Matrix(self.output_nodes, 1)
            self.bias_h.randomize()
            self.bias_o.randomize()

            self.set_learning_rate()
            self.set_activation_function()

        def predict(self, input_array):
            inputs = Matrix.from_array(input_array)
            hidden = Matrix.static_multiply(self.weights_ih, inputs)
            hidden.add(self.bias_h)
            hidden.map(self.activation_function.func)

            output = Matrix.static_multiply(self.weights_ho, hidden)
            output.add(self.bias_o)
            output.map(self.activation_function.func)

            return output.to_array()

        def set_learning_rate(self, learning_rate=0.1):
            self.learning_rate = learning_rate

        def set_activation_function(self, func=sigmoid):
            self.activation_function = func

        def train(self, input_array, target_array):
            inputs = Matrix.from_array(input_array)
            hidden = Matrix.static_multiply(self.weights_ih, inputs)
            hidden.add(self.bias_h)
            hidden.map(self.activation_function.func)

            outputs = Matrix.static_multiply(self.weights_ho, hidden)
            outputs.add(self.bias_o)
            outputs.map(self.activation_function.func)

            targets = Matrix.from_array(target_array)

            output_errors = Matrix.subtract(targets, outputs)

            gradients = Matrix.static_map(outputs, self.activation_function.dfunc)
            gradients.multiply(output_errors)
            gradients.multiply(self.learning_rate)

            hidden_T = Matrix.transpose(hidden)
            weight_ho_deltas = Matrix.static_multiply(gradients, hidden_T)
            self.weights_ho.add(weight_ho_deltas)
            self.bias_o.add(gradients)

            who_T = Matrix.transpose(self.weights_ho)
            hidden_errors = Matrix.static_multiply(who_T, output_errors)
            hidden_gradient = Matrix.static_map(hidden, self.activation_function.dfunc)
            hidden_gradient.multiply(hidden_errors)
            hidden_gradient.multiply(self.learning_rate)

            inputs_T = Matrix.transpose(inputs)
            weight_ih_deltas = Matrix.static_multiply(hidden_gradient, inputs_T)
            self.weights_ih.add(weight_ih_deltas)
            self.bias_h.add(hidden_gradient)

        def copy(self):
            return NeuralNetwork(self)

        def mutate(func):
            self.weights_ih.map(func)
            self.weights_ho.map(func)
            self.bias_h.map(func)
            self.biah_o.map(func)
