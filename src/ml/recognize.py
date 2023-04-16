import numpy
import scipy.special
from PIL import Image


class NeuralNetwork:
    """ Описание сети, содержащей три слоя: входной, скрытый и выходной """

    def __init__(self,
                 input_nodes: int,
                 hidden_nodes: int,
                 output_nodes: int,
                 learning_rate: float) -> None:
        """ 
        Конструктор создает сеть со случайными значениями весовых коэффициентов.

        :param input_nodes: количество нейронов во входном слое
        :param hidden_nodes: количество нейронов во входном слое
        :param output_nodes: количество нейронов во входном слое
        :param learning_rate: количество нейронов во входном слое
        """
        self._input_nodes = input_nodes
        self._hidden_nodes = hidden_nodes
        self._output_nodes = output_nodes
        self._learning_rate = learning_rate

        self._weights_input_hidden = numpy.random.normal(
            0.0,
            pow(self.hidden_nodes, -0.5),
            (self.hidden_nodes, self.input_nodes),
        )
        self._weights_hidden_output = numpy.random.normal(
            0.0,
            pow(self.output_nodes, -0.5),
            (self.output_nodes, self.hidden_nodes),
        )

        self._actiivation_function = lambda x: scipy.special.expit(x)

    @property
    def input_nodes(self) -> int:
        return self._input_nodes

    @property
    def hidden_nodes(self) -> int:
        return self._hidden_nodes

    @property
    def output_nodes(self) -> int:
        return self._output_nodes

    @property
    def learning_rate(self) -> int:
        return self._learning_rate

    @property
    def weights_input_hidden(self) -> int:
        return self._weights_input_hidden

    @property
    def weights_hidden_output(self) -> int:
        return self._weights_hidden_output

    def predict(self, inputs_list):
        input_vector = numpy.array(inputs_list).T

        hidden_inputs = numpy.dot(self._weights_input_hidden, input_vector)
        hidden_outputs = self._actiivation_function(hidden_inputs)

        final_inputs = numpy.dot(self._weights_hidden_output, hidden_outputs)
        final_outputs = self._actiivation_function(final_inputs)

        return final_outputs


def recognize_digits(image: Image.Image):
    nn = NeuralNetwork(
        input_nodes=3,
        hidden_nodes=2,
        output_nodes=3,
        learning_rate=0.3,
    )

    print(nn)
    print(nn.weights_input_hidden)
    print(nn.weights_hidden_output)

    out = nn.predict([10, 20, 30])

    # Здесь будет код ML для распознавания цифр
    return str(out)
