# Deep Dive Proposal: Actor-Critic Reinforcement Learning
## Project Overview
### Project Goals
For this project, I plan to implement an Actor-Critic algorithm to train a mobile robotic agent to navigate a simulated environment warehouse. I will build on my learning from the midterm project, in which I implemented Q-Learning for navigation of a simplified warehouse environment. For this project, I will first design and implement a continuous, stochastic warehouse environment containing obstacles and a destination for the agent to reach. Next, I will implement the Advantage Actor Critic (A2C) algorithm and train the agent to find the optimal path from its starting position to its goal. Finally, I will benchmark the algorithm's rate of convergence given varying parameters.
### What Are Actor-Critic Algorithms?
Actor-Critic algorithms are a family of algorithms that combine value-based and policy-based methods of reinforcement learning. Value-based methods, including Q-Learning, do not attempt to approximate a specific policy of actions to take; rather, they estimate the reward associated with given state-action pairs. On the other hand, policy-based methods do directly approximate a value function by tuning parameters of the policy function to find an optimal solution. In Actor-Critic algorithms, the Actor takes actions using a policy-based method, and the Critic evaluates those actions using a value-based method. Since value-based algorithms depend on the action taken, the Critic must learn alongside the Actor in order to successfully critique its actions.
### Motivation and Learning Goals
Through my midterm project on Q-Learning, I strengthened my familiarity with reinforcement learning concepts. Through this project, I'd like to advance my knowledge by working with an algorithm that is cutting-edge in the RL world and broadly applicable to robotics. I particularly like the Actor-Critic algorithm family because it allows me to work with both value-based and policy-based methods of RL, which will give me a strong foundation in the mathematics and significance of each.

An additional learning goal of mine is to practice developing a high-quality, low-fidelity simulation environment. My last environment was simple and discrete in nature; this time I'd like to build a continuous environment with inherent stochasticity. This will allow me to highlight the strengths of Actor-Critic algorithms while also giving me the chance to build my own robust test environment.

Finally, I'd like to exercise my ability to scope and execute a project that I entirely own. As such, I plan to work independently on this project.
## Deliverables
As a final deliverable for this project, I plan to write a final report on my learning that includes the following components:
### Summary of Q-Learning
My report will begin with a discussion of how Actor-Critic algorithms work. I plan to include equations and pseudocode that articulate the mathematical basis of the Actor and Critic components. I will articulate why Actor-Critic algorithms are more powerful than value-based or policy-based methods alone, and if they have any significant downsides. I will also articulate the specifications required to implement A2C specifically.
### Codebase of Q-Learning Implementation
I will implement the Advantage Actor Critic (A2C) algorithm to train a robotic agent to find the optimal path to a destination in a model warehouse environment.

I envision a warehouse world model in which a robotic agent has a starting position and a goal position in continuous 2D space. The world will contain impassible obstacles (such as shelves) and/or areas of discouraged travel (such as highly-trafficked zones). These areas will incur a cost if the robot traverses  them, which represents the overall reward field. In addition to static obstacles, there may also be dynamic/moving obstacles, representing humans, forklifts, or even other robots.

The robotic agent will be able to execute linear and angular velocity commands, just like a real differential-drive robot. The agent chooses an action using its Actor, which learns the policy function to optimally map states to actions. The Critic then evaluates that state-action pair by approximating the value function that maps the pair to the reward the agent receives for it. The resultant q-value is used by the Actor to refine its parameters for the next choice of action.

In addition to the world and the robotic agent, I plan to have a class dedicated to visualizing each step of the learning process. The visualizer will create animated plots of the agent as it explores the world model, as well as benchmarking plots that summarize how quickly the algorithm can find the optimal path given variations on the same environment.
### Performance Analysis
I will analyze the success of the A2C algorithm given variations on the warehouse environment.  Given the plots created by my visualizer class, I will be able to explain why the algorithm performs better or worse in different scenarios. I will propose possible improvements to the algorithm, and/or speculate on how different methods of RL might compare to the performance of my algorithm implementation. This analysis will be in the form of a final report, alongside my initial summary of the algorithm.
## Rubric
I propose the following grading scheme, in which I receive up to 8 points for completing the project:

A: 7-8 pts | B: 5-6pts | C: 3-4pts | D = 1-2pts | F: 0pts

- Algorithm summary: 1 point
- Warehouse model implementation: 1 point
- Algorithm implementation: 4 points
- Plots and analysis: 1 point
- Unit testing: 1 point
## Timeline
I will commit to the following timeline for my project, given that I have three weeks to complete it.
### Week 1 (11/17 - 11/24): Algorithm Summary and Environment Model
I will start by reading about the A2C algorithm and its underlying math. I will compile resources I find on the topic and synthesize them such that I'm prepared to confidently implement the algorithm. I will demonstrate my learning by writing the A2C Algorithm Summary component of my final report, including pseudocode and equations to describe the algorithm.

At the same time, I will implement my warehouse environment as a continuous reward field with inherent stochasticity in its state transitions. I will utilize the Pymunk or PyBox2D libraries to help me in this process. I will ensure that my environment is robust by writing a thorough suite of unit tests to accompany it. If I'm still struggling to get this working by the end of the week, I will pivot to OpenAI's RL Gymnasium library instead to model my environment.
### Checkpoint 1 (11/20)
Checkpoint 1 occurs halfway through the first week of the project. By then, I plan to have the Algorithm Summary portion of my report completely finished. I also plan to have an Interface describing the necessary functions and properties of my Environment class, even if they aren't yet implemented.
### Week 2: (12/1 - 12/8) Algorithm Implementation and Testing
I will implement the Actor and Critic of the A2C algorithm. I will go into this process with a strong grasp on the underlying math, so the biggest challenge will be implementing the equations programmatically and connecting the theoretical states and actions to the existing environment I've built. I hope to unit test as much of this class as possible.
### Checkpoint 2 (12/8)
Checkpoint 2 occurs at the end of the second week of the project. By then, I plan to have broadly completed the technical components of my project. This includes a robust continuous warehouse environment, and a robotic agent that learns to optimally traverse that environment using the A2C algorithm.
### Week 3: Benchmarking, Visualizations, and Contingency
I will create visualizations of the training in action so that the progress from the first episode to the last is easy to understand. If there is time, I'll adjust the parameters of the algorithm to see if I can optimize performance given a certain stochasticity or obstacle density in the environment. Finally, I will set this week aside for any catch-up work that is left to do.
