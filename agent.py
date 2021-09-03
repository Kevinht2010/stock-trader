import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from buffer import Buffer
from loss import loss_fun


class Agent:

    def __init__(self, window_size=14, reset_every=1000, pretrained=False, model_name=None) -> None:
        self.model_name = model_name
        self.action_size = 3
        self.inventory = []
        self.memory = Buffer(1000, (window_size, 2))
        self.state_shape = (window_size, 2)
        self.window_size = window_size

        self.first_iter = True
        self.gamma = 0.93
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.loss_fun = loss_fun
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )

        self.custom_objects = {"loss": loss_fun}

        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self.create_model()

        self.n_iter = 1
        self.reset_every = reset_every

        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        lstm_layer_1 = tf.keras.layers.LSTM(
            units=64,
            input_shape=self.state_shape,
            return_sequences=True,
            activation="tanh", dropout=0.1,
        )
        lstm_layer_2 = tf.keras.layers.LSTM(
            units=32,
            activation="tanh", dropout=0.1,
        )
        dense_layer = tf.keras.layers.Dense(
            units=self.action_size,
        )
        inputs = tf.keras.Input(shape=self.state_shape)
        x = inputs
        x = lstm_layer_1(x)
        x = lstm_layer_2(x)
        x = dense_layer(x)
        outputs = x
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss=self.loss_fun,
            optimizer=self.optimizer,
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.remember(state, action, reward, next_state, done)

    def act(self, state, is_eval=False):
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            return 1  # make a definite buy on the first iter

        action_probs = self.model.predict(state)
        return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        if self.memory.memory_counter % self.reset_every == 0:
            self.target_model.set_weights(self.model.get_weights())

        states, actions, rewards, next_states, dones = self.memory.get_batch(
            32)

        indices = np.arange(batch_size, dtype=np.int32)

        dones = [1 if d == 0 else 0 for d in dones]
        action_indices = [np.argmax(k)
                          for k in self.model.predict(next_states)]

        target = rewards + self.gamma * \
            self.target_model.predict(next_states)[
                indices, action_indices] * dones

        q_values = self.model.predict(states)
        q_values[indices, actions] = target

        loss = self.model.fit(
            np.array(states), np.array(q_values),
            epochs=1, verbose=0
        ).history["loss"][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        self.model.save("models/{}".format(self.model_name))

    def load(self):
        return load_model("models/" + self.model_name, custom_objects=self.custom_objects)
