# TEACHING (or trying to) A NEURAL NETWORK TO PLAY TIC TAC TOE!

This is a project for the discipline of Artificial Inteligence I'm taking at university
The assignment was to develop a game and teach a neural network to play it using both: backpropagation & genetic algorithm.

For this project, I have used:
### Min-max algorithm for generating the dataset
This algorithm consists basically in testing exhaustively all the possible moves in the board and check if that move could lead to a victory.
As there could be more than one optimal moviment, I thought it would be a good idea to evaluate the model checking if the most likely optimal moviment (the network output) is present in this group.
This could happen specially during the first rounds, when many different movements could lead the model to a victory.
### Backpropagation
Implementes the forward and backward-passes by hand, I am using pytorch only because it makes it much easier to multiply and debug matrices.
Also, this is a very simple implementation of Stochastic Gradient Descent, without normalizations, residual conections, etc.
### Genetic algorithm (required by the assignment)
While I was planning the game, I thought on discarding the cross-over step and keep only the mutation and evaluation, and I think it makes sense for this problem.
For the mutation, I thought about three different ways to add noise to the network:
1. Absolute variation
    This won't consider the weights direction nor magnitude, it's simply a
    w + (torch.rand_like(w) - 0.5)
2. Standard Deviation + noise
    The idea is to use the STD of the layer to control the mutation.
    It considers how far a weight is from the mean of that layer and apply a random noise on each neuron based on that.
    Therefore, neurons close to the mean of the layer would barely move, but the distant points could get closer or more distant from the average.
    I believe this is worth testing specially when using residual connection, today, this network does f(x) = wx + b.
    w + (torch.rand_like(w) * std(w))
3. LayerNorm
    The idea is to use the normalization of the layer to control the mutation as well.
    The mutation consists, basically, in: w + (LayerNorm(w) * w)

## Instructions for cloning and running

> git clone git@github.com:joaoPedrooC/ai-tic-tac-toe.git

or

> git clone https://github.com/joaoPedrooC/ai-tic-tac-toe.git

Installing the dependencies (on proj. dir)

> python -m pip install -r requirements.txt

then

> python -m train.train.py

[NOTE]: I haven't finished the project nor the documentation<br>
I'll be updating it from time to time, if you want to discuss something about this project, feel free to contact me<br>
My socials are on my profile! :)