from tensorforce import Configuration
from tensorforce.agents import *
from tensorforce.agents import PPOAgent
from planestrike_env import PlaneStrike
from tensorforce.execution import Runner
from tensorforce.core.networks import layered_network_builder

import numpy as np
import pylab

ts = []

env = PlaneStrike()

agents = []
agent = VPGAgent(config=Configuration(
    loglevel='debug',
    batch_size=1,
    states=env.states,
    actions=env.actions,
    network=layered_network_builder([
        dict(type='dense', size=32),
        dict(type='dense', size=32),
    ]),
))
agents.append(agent)

agent = TRPOAgent(config=Configuration(
    loglevel='info',
    batch_size=32,
    # generalized_advantage_estimation=True,
    # normalize_advantage=False,
    # gae_lambda=0.97,
    # override_line_search=False,
    # cg_iterations=20,
    # cg_damping=0.01,
    # line_search_steps=20,
    # max_kl_divergence=0.005,
    states=env.states,
    actions=env.actions,
    network=layered_network_builder([
        dict(type='dense', size=32),
        dict(type='dense', size=32)
    ])
))
agents.append(agent)

agent = PPOAgent(config=Configuration(
    loglevel='debug',
    batch_size=32,
    states=env.states,
    actions=env.actions,
    optimizer_batch_size=10,
    network=layered_network_builder([
        dict(type='dense', size=32),
        dict(type='dense', size=32),
    ]),
))
agents.append(agent)

agent = DQNAgent(config=Configuration(
    loglevel='debug',
    batch_size=32,
    states=env.states,
    actions=env.actions,
    target_update_frequency=1000,
    double_dqn=True,
    network=layered_network_builder([
        dict(type='dense', size=32),
        dict(type='dense', size=32)
    ])
  # preprocessing=None,
  # # exploration={
  # #   type:'epsilon_decay',
  # #   epsilon:1.0,
  # #   # epsilon_final=0.1,
  # #   # epsilon_timesteps=1e6
  # # },
  #
  # memory_capacity=10000,
  # memory='replay',
  # update_frequency=4,
  # first_update=50000,
  # repeat_update=1,
  # discount=0.97,
  # learning_rate=0.00025,
  # # optimizer={
  # #   dict(type='rmsprop'),
  # #   dict(momentum=0.95),
  # #   dict(epsilon=0.01)
  # # },
  # tf_saver=False,
  # tf_summary=None,
  #
  # update_target_weight=1.0,
  # clip_gradients=0.0,
))
agents.append(agent)



# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.timestep,
                                                                                 reward=r.episode_rewards[-1]))
    ts.append(r.timestep)
    return True

for agent in agents:
    # Create the runner
    runner = Runner(agent=agent, environment=env)
    # Start learning
    runner.run(episodes=1000, max_timesteps=50, episode_finished=episode_finished)

    # Print statistics
    print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(ep=runner.episode,
                                                                                                       ar=np.mean(
                                                                                                           runner.episode_rewards[
                                                                                                       -100:])))
    WINDOW_SIZE = 50
    tmp = [np.mean(runner.episode_rewards[i:i + WINDOW_SIZE]) for i in range(len(runner.episode_rewards) - WINDOW_SIZE)]
    pylab.plot(tmp, label=agent.name.replace('Agent', ''))
    # tmp = [np.mean(ts[i:i + WINDOW_SIZE]) for i in range(len(ts) - WINDOW_SIZE)]
    # pylab.plot(ts)

pylab.legend(loc='lower right')
pylab.show()
