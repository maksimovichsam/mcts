# Learning to play TicTacToe

In my first implementation of alpha zero, I didn't have a metric for if my tictactoe network was actually learning. The way I would discern if my algorithm worked was just by playing the network by hand and seeing if I could find a blind spot, which I considered the network to be a failure. This was very difficult and I ended up giving up since I didn't know what was wrong with my algorithm.

I came back a few months later and decided to setup a metric for if my network is actually learning. To do this, I used the minimax algorithm to solve tictactoe and generate a list of optimal moves from every position in the game. Surprisingly, this only resulted in 4520 states. Anyways, the original network I trained gave 23% accuracy from this dataset. I spent a long time trying to figure out why. What I ended up doing was running my alpha-zero code and a public github version side by side with a debugger and find out where they diverged. This was a difficult and tedious process, but here is what I found I was doing wrong:

1. I was representing the board state using 3x3 board, where each cell in the board was a one hot encoded vector of size 3 (i.e [0, 0, 0] for nothing, [0, 1, 0] for X, etc.). The public alpha-zero code has a function called `getCanonicalBoard`, which returns the board in a player independent way. For example, they use a 3x3 board where each cell is 0 for nothing, 1 for the current player, -1 for the opponent. This means that both X's and O's can be represented as 1's. 

2. There was a bug in my code where I would be accidentally resetting the game tree every time I reached a node in the game tree that had already existed. 

3. In computing the loss between probability vectors, I was using cross entropy between probability vectors masked by the current legal actions of that state. This made me have to loop over each example since each probability vector was a different length. I'm not sure why this made I big difference, I would think the loss would be the same regardless.

After solving these my network reached an accuracy of around 83%. 

## Fine tunning

Here are a few things I did to increase this to 97%. It's important to note that I did not want to use the natural symmetry of the game to augment the training data. I didn't want to do this because it felt like a cheat, and I wanted to reproduce the alpha zero paper as much as possible.

1. I used Bayesian Optimization to find a better value of c_puct. I found 4 worked very well for me. This increased my accuracy to 88%. Random search would have been fine here too, honestly.
   
2. Add Dirichlet noise to the root node in MCTS simulations. I use alpha=1 and epsilon=0.25. This increased my accuracy to 90%

3. I repeatedly re trained models until their accuracy increased. For example, I trained the 90% model with more simulations and lower learning rate. this increased it to 94, then 95, 96, 97. 

## Questions I don't have the answers to

1. A supervised model was able to achieve 99% accuracy, but the alpha zero was only able to learn 96 after a lot of fine tuning. What's holding back the model from reaching 99% in the same way?
