import numpy as np


class Buffer:

    def __init__(self, mem_sz=1000, state_shape=(14, 2)) -> None:
        self.memory_counter = 0
        self.memory_size = mem_sz
        self.state_shape = (mem_sz,) + state_shape
        self.state_memory = np.zeros(self.state_shape)
        self.new_state_memory = np.zeros(self.state_shape)

        self.action_memory = np.zeros(self.memory_size, dtype=np.int8)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.float32)

    def get_batch(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal

    def remember(self, state, action, reward, next_state, done):
        index = self.memory_counter % self.memory_size

        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.memory_counter += 1
