from random import sample

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Flatten, Dense, Activation
import numpy as np


def make(env):
    return Agent(env.observation_space.shape[0], env.action_space.n)


def build_net(num_inputs, num_outputs):
    out = entry = Input(shape=[num_inputs])
    out = Dense(14, activation='relu')(out)
    out = Dense(14, activation='relu')(out)
    out = Dense(num_outputs)(out)

    net = Model(entry, out)
    net.compile(loss='mse', optimizer=Adam(lr=0.01, decay=0.004))
    return net

def predict(net, x):
    return net.predict_on_batch(np.expand_dims(x, 0))[0]

class Agent:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_episodes = 0

        self.net1 = build_net(num_states, num_actions)
        self.net2 = build_net(num_states, num_actions)

        self.history = []
        self.state, self.action = None, None

        self.batch_size = 64
        self.gamma = 0.9
        self.epsilon, self.epsilon_rate = 1.0, 0.99
        self.history_size = 128

    def decide_action(self, state):
        actions = range(self.num_actions)
        if np.random.rand() >= self.epsilon:
            q = predict(self.net1, state)
            q_max = q.max()
            actions = [a for a in actions if q[a] == q_max]
        action = np.random.choice(actions)

        self.state, self.action = state, action
        return action

    def receive_feedback(self, reward, state_new, done):
        self.history.append(
            (self.state, self.action, reward, None if done else state_new))
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

        if done:
            self.num_episodes += 1
            self.epsilon *= self.epsilon_rate
            self.replay()
            if self.num_episodes % 10 == 0:
                self.net1.set_weights(self.net2.get_weights())

    def replay(self):
        if len(self.history) < self.batch_size:
            return

        x, y = [], []
        for state, action, reward, state_new in sample(self.history, self.batch_size):
            target = predict(self.net2, state)
            target[action] = reward
            if state_new is not None:
                target[action] += self.gamma * predict(self.net1, state_new).max()
            x.append(state), y.append(target)

        self.net2.train_on_batch(np.stack(x), np.stack(y))
