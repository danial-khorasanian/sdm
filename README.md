# Sequential Decision-Making Optimization

This repository provides the codes for some of the examples given in the "Sequential Decision-Making Optimization" chapter within the Book "[Creative Artificial Intelligence for Discovery Automation](https://www.cai4discovery.com)" written by Danial Khorasanian. 

There are three main sections:
- Dynamic Programming,
- Monte Carlo Control, and
- Reinforcement Learning.

## Reinforcement Learning
RL methods and the problems solved with each include:
- Q-learning: Maze
- DQN
    - Car
    - Pong
- Reinforce: [Inverted Pendulum](environments/inverted_pendulum/experiments.ipynb)
- Advantage Actor-Critic (A2C): [Inverted Pendulum](environments/inverted_pendulum/experiments.ipynb)
- Proximal Policy Optimization (PPO): [Inverted Pendulum](environments/inverted_pendulum/experiments.ipynb)

## Training Results

<table border="0" width="100%">
  <tr>
    <td width="20%" align="center">
      Maze (Q-learning)
    </td>
    <td width="2.5%" align="center">
      &nbsp;
    </td>
    <td width="35%" align="center">
      <img src="environments/maze/plots/maze_ql.png" height="220"/>
    </td>
    <td width="2.5%" align="center">
      &nbsp;
    </td>
    <td width="30%" align="center">
      <img src="environments/maze/gifs/maze_ql.gif" height="240"/>
    </td>
  </tr>

  <tr>
    <td align="center">
      Car (DQN)
    </td>
    <td align="center">
      &nbsp;
    </td>
    <td align="center">
      <img src="environments/car/plots/car_dqn_episode_returns.png" height="230"/>
    </td>
    <td align="center">
      &nbsp;
    </td>
    <td align="center">
      <img src="environments/car/gifs/car_dqn.gif" height="200"/>
    </td>
  </tr>

  <tr>
    <td align="center">
      Pong (DQN)
    </td>
    <td align="center">
      &nbsp;
    </td>
    <td align="center">
      <img src="environments/pong/plots/pong_dqn_episode_returns.png" height="230"/>
    </td>
    <td align="center">
      &nbsp;
    </td>
    <td align="center">
      <img src="environments/pong/gifs/pong_dqn.gif" height="250"/>
    </td>
  </tr>

  <tr>
    <td align="center">
      Inverted Pendulum (REINFORCE, A2C, PPO)
    </td>
    <td align="center">
      &nbsp;
    </td>
    <td align="center">
      <img src="environments/inverted_pendulum/plots/inv_pend_all_agents.png" height="230"/>
    </td>
    <td align="center">
      &nbsp;
    </td>
    <td align="center">
      <img src="environments/inverted_pendulum/gifs/invpend_reinforce.gif" height="220"/>
    </td>
  </tr>
</table>