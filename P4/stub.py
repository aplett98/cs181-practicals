# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self, gamma=0.6, epsilon=0.1, eta=0.5):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

        # hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.eta = eta

        # linear approximation setup
        self.d = 8  # dim(S)=7, dim(A)=1 -> dim(S x A) = 8
        self.w = np.ones(self.d)  # weights for linear approximation

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.gravity = 2

    def _phi(self, state, action):
        '''This will be the basis function for linear approximation.'''
        return np.array(
            [
                action,
                self.gravity,
                state["score"],
                state["tree"]["dist"],
                state["tree"]["top"],
                state["tree"]["bot"],
                state["monkey"]["vel"],
                state["monkey"]["top"]
            ]
        )

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the
        # last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        new_action = npr.rand() < 0.1
        new_state = state

        self.last_action = new_action
        self.last_state = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    '''
    Driver function to simulate learning by having the agent play
    a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(
            sound=False,             # Don't play sounds.
            text=f"Epoch {ii}",      # Display the epoch on screen.
            tick_length=t_len,       # Make game ticks super fast.
            action_callback=learner.action_callback,
            reward_callback=learner.reward_callback
        )

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games.
    run_games(agent, hist, 20, 10)

    # Save history.
    np.save('hist', np.array(hist))
