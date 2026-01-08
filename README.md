# Advantage Actor-Critic for Lunar Landing

> Ivy Mahncke\
> DSA Fall 2025 Final Project

## Project Overview

This is my final project for Data Structures & Algorithms, Fall 2025. In this project, I was challenged to study and implement an algorithm of my choice. I chose to explore actor-critic algorithms, a family of deep reinforcement learning algorithms that employ value-based and policy-based methods to find optimal behaviors. To demonstrate my learning, I trained a robotic agent to solve the Gymnasium Lunar Lander environment using the Advantage Actor-Critic algorithm!

To learn more, please check out the [final report I wrote about the project,](docs/report.md) where I cover everything I did in a lot more detail!

My [initial project proposal document](docs/proposal.md) can also be found in this repository.

To see the assignment description and expectations, feel free to do so on [our class website.](https://olindsa2025.github.io/assignments/assignment_09.html)

## File Structure

This project contains the following files:

```bash
├── src
│   ├── a2c.py
│   ├── lander.py
├── docs
│   ├── proposal.md
│   ├── report.md
│   ├── lunar_lander.gif
├── requirements.txt
├── README.md
```

`src` houses the source code for this project. Here's a quick descriptor of the contents:
- `a2c.py`: my implementation of the Advantage Actor-Critic algorithm, which uses PyTorch neural networks to model the actor and critic.
- `lander.py`: the main script for training the lunar lander agent using A2C.

`docs` houses all documentation of my project. The two important files in here are:
- `proposal.md`: my project proposal! It contains learning goals, planned deliverables, and a project timeline.
- `report.md`: my final report! It contains my background research, methodology, and results, as well as ideas for future work.
- `lunar_lander.gif`: a helpful visual of the training environment and the lunar lander agent :)
