import numpy
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


def recognize_digits(image: Image.Image):
    nn = NeuralNetwork(
        input_nodes=3,
        hidden_nodes=3,
        output_nodes=3,
        learning_rate=0.3,
    )

    print(nn)

    # Здесь будет код ML для распознавания цифр
    return 42
