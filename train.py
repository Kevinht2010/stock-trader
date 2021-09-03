import os
import logging
import math
import numpy as np

from utils import sigmoid
from utils import get_state
from utils import format_1, format_2


def train_model(agent, episode, data,
                ep_count=100, batch_size=32, window_size=10):
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []

    state = get_state(data, 0, window_size + 1)

    for t in range(data_length):
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)
        action = agent.act(state)

        if action == 1:
            agent.inventory.append(data[t])

        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t][0] - bought_price[0]
            reward = 100*(delta/bought_price[0])
            total_profit += delta

        else:
            pass

        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        if agent.memory.memory_counter > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 3 == 0:
        agent.save(episode)

    return {
        'episode': episode,
        'ep_count': ep_count,
        'total profit': total_profit,
        'mean loss': np.mean(np.array(avg_loss)),
    }
