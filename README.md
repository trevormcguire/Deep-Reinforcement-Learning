# Deep-Reinforcement-Learning
Deep Reinforcement Learning


This project is purposed towards using Deep Q Learning with action and target networks to learn when to buy and sell stocks. The buying and selling tasks are split between two different agents, each with their own discount rate (gamma). 

For rewards, Test1.py uses a Profit Per Tick (PPT) Weighted Mean method, which--in conjuction with the low gamma rate--discourages the Seller Agent to hold for long periods of time.

The buyer, however, has a relatively significant gamma rate (0.6) in Test1.py, to teach the agent to look for only the best opportunities. 
