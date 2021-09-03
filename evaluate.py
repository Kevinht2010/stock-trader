import os
import logging
import math
import numpy as np

from utils import get_state
from utils import format_1, format_2


def evaluate_model(agent, data, window_size, debug):
    """
        Arguments:
            agent: the agent.
            data: the data used to evaluate the model.
            window_size (int): the number of time units taken into account to do a prediction.
            debug (bool): controls whether or not to debug.
        Returns:
            a tuple of length 2 consisting of total profit and trading history.
                total profit is a float.
                trading history is an array of price-action pairs.
                    an action is one of the strings "BUY", "SELL", or "HOLD".
    """

    total_profit = 0

    history = []
    agent.inventory = []

    state = get_state(data, 0, window_size + 1)
    data_length = len(data) - 1

    for t in range(data_length):
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)

        action = agent.act(state, is_eval=True)

        if action == 1:
            agent.inventory.append(data[t])

            history.append((data[t][0], "BUY"))
            if debug:
                logging.debug("Buy at: {}".format(format_1(data[t][0])))

        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t][0] - bought_price[0]
            reward = delta
            reward = 100*(delta/bought_price[0])
            total_profit += delta

            history.append((data[t][0], "SELL"))
            if debug:
                logging.debug("Sell at: {} | Position: {}".format(
                    format_1(data[t][0]), format_2(data[t][0] - bought_price[0])))

        else:
            history.append((data[t][0], "HOLD"))
            if debug:
                logging.debug("Hold at: {}".format(format_1(data[t][0])))

        agent.memory.remember(state, action, reward,
                              next_state, (t == data_length - 1))
        state = next_state

    return {
        'total profit': total_profit,
        'trading history': np.array(history),
    }
