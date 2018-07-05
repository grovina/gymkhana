import gym
import agent
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go


env = gym.make('CartPole-v0')
agt = agent.make(env)

rewards = []
for episode in range(1001):
    state = env.reset()
    done = False
    rewards.append(0)
    while not done:
        if episode % 100 == 0: env.render()
        action = agt.decide_action(state)
        state, reward, done, _ = env.step(action)
        agt.receive_feedback(reward, state, done)
        rewards[-1] += reward

    if episode % 100 == 0:
        average = np.mean(rewards[-100:])
        print("episode {}: average {}".format(episode, average))    

env.close()

py.plot([go.Scatter(y=rewards)], filename='rewards.html')
