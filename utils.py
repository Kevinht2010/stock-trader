import numpy as np
import os
import math
import logging

import tensorflow.keras.backend as K


def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


def format_1(price):
    return ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))


def format_2(price):
    return '${0:.2f}'.format(abs(price))


def get_state(data, t, n_days):
    d = t - n_days + 1

    if d >= 0:
        block = data[d: t + 1]
    else:
        block = -d * [data[0]] + data[0: t + 1]

    res = []
    for i in range(n_days - 1):
        res.append((sigmoid(100*(block[i + 1][0] - block[i][0])/block[i][0]),
                    sigmoid(100*(block[i + 1][1] - block[i][1])/block[i][1])))
    return np.array([res])


def switch_k_backend_device():
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
