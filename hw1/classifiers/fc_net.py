from builtins import object
import os
import numpy as np

from .layers import *
from .layer_utils import *


class TwoLayerNet(object):
    """
    Двухслойная полносвязная нейронная сеть с нелинейностью ReLU и функцией потерь softmax, 
    использующая модульную структуру слоев. Мы предполагаем размерность входных данных D, 
    размерность скрытого слоя H и выполняем классификацию по C классам.

Архитектура должна быть полносвязный слой - reLU - полносвязный слой - softmax.

Обратите внимание, что этот класс не реализует градиентный спуск; вместо этого он
будет взаимодействовать с отдельным объектом Solver, который отвечает за выполнение
оптимизации.

Обучаемые параметры модели хранятся в словаре
self.params, который сопоставляет имена параметров с массивами numpy.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Инициализация.

        Входы:
        - input_dim: Целое число, указывающее размер входных данных
- hidden_dim: Целое число, указывающее размер скрытого слоя
- num_classes: Целое число, указывающее количество классов для классификации
- weight_scale: Скаляр, указывающий стандартное отклонение для случайной инициализации весов.
- reg: Скаляр, указывающий силу L2-регуляризации.
        """
        self.params = {}
        self.reg = reg

        self.params = {
          'W1' : weight_scale * np.random.randn(input_dim, hidden_dim),
          'b1' : np.zeros(hidden_dim), 
          'W2' : weight_scale * np.random.randn(hidden_dim, num_classes),
          'b2' : np.zeros(num_classes)
        }

    def loss(self, X, y=None):
        """
        Вычисляет функцию потерь и градиент для мини-батча данных.
Входные данные:
- X: Массив входных данных формы (N, d_1, ..., d_k)
- y: Массив меток формы (N, ..., d_k). y[i] — метка для X[i].

Возвращает:
Если y равно None, то выполняется прямой проход модели во время тестирования и возвращается:
- scores: Массив формы (N, C), содержащий оценки классификации, где
scores[i, c] — оценка классификации для X[i] и класса c.
Если y не равно None, то выполняется прямой и обратный проходы во время обучения и
возвращается кортеж из:
- loss: Скалярное значение, определяющее функцию потерь
- grads: Словарь с теми же ключами, что и self.params, сопоставляющий имена параметров
с градиентами функции потерь относительно этих параметров.
        """
        scores = None
        W1 = self.params["W1"]
        b1 = self.params["b1"]
        W2 = self.params["W2"]
        b2 = self.params["b2"]

        out1, cache1 = affine_relu_forward(X, W1, b1)
        scores, cache2 = affine_forward(out1, W2, b2)

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
      
        loss, dloss = softmax_loss(scores, y)
        # Убедимся что реализация включает L2 регуляризацию 
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))
        # далее обновим параметры с новыми новыми значениями градиентов

        dout, dW2, db2 = affine_backward(dloss, cache2)
        dW2 += self.reg * W2
        dX, dW1, db1 = affine_relu_backward(dout, cache1)
        dW1 += self.reg * W1

        grads = {
            'W1' : dW1,
            'b1' : db1,
            'W2' : dW2,
            'b2' : db2
        }

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "..", "saved", fname) #так как запускаю на windows, другой слеш
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "..", "saved", fname) #так как запускаю на windows, другой слеш
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True

