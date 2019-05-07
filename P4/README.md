# Practical 4
All code is in `stub.py`. To run the learner that gave us our high score of ~120, run as-is.
There are also modifiable hyper-parameters:
`gamma`: Decay (from Q-learning)
`epsilon`: Epsilon for the epsilon-greedy approach. Decays linearly.
`eta`: Learning rate (from Q-learning)
`xbins`: The number of bins for approximating x-distance to the next tree in the state space.
`ybins`: The number of bins for approximating y-distance to the next tree gap in the state space.
These can be changed directly as arguments when creating an instance of `Learner`.
The scores for the last run are saved in `hist.npy`.
