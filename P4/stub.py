# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self, gamma=0.4, epsilon=0.5, eta=0.91):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.gravity = 1
        self.ginit = False

        # hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.eta = eta

        # q-learning setup
        self.ybins = np.linspace(-150, 150, 5, dtype=int)
        self.xbins = np.linspace(0, 500, 8, dtype=int)
        self.gbins = np.array([1, 4])
        self.vbins = np.linspace(-30, 30, 3, dtype=int)
        self.lookup = {}
        i = 0
        for y in self.ybins:
            for x in self.xbins:
                for g in self.gbins:
                    for v in self.vbins:
                        self.lookup[(y, x, g, v)] = i
                        i += 1
        self.w = list(npr.rand(i, 2))
        print(np.array(self.w).shape)

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.gravity = 1
        self.ginit = False
        if self.epsilon > 0:
            self.epsilon -= 0.01

    def disc(self, state):
        '''This will be the basis function for linear approximation.'''
        g = self.gravity
        x = np.clip(
            int(np.digitize(state["tree"]["dist"], self.xbins)),
            0, len(self.xbins) - 1)
        x = self.xbins[x]
        y = np.clip(int(np.digitize(
            state["tree"]["top"] - state["monkey"]["top"], self.ybins
        )), 0, len(self.ybins) - 1)
        v = np.clip(int(np.digitize(
            state["monkey"]["vel"], self.vbins
        )), 0, len(self.vbins) - 1)
        v = self.vbins[v]
        y = self.ybins[y]
        return self.lookup[(y, x, g, v)]

    def Q(self, state, action):
        return self.w[self.disc(state)][action]

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the
        # last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        if self.last_state is None:
            self.last_action = (
                0 if self.Q(state, 0)
                >= self.Q(state, 1) else 1
            )
            self.last_state = state
            return self.last_action

        if (
            state["monkey"]["vel"] < self.last_state["monkey"]["vel"]
            and not self.ginit
        ):
            self.gravity = (
                abs(state["monkey"]["vel"] - self.last_state["monkey"]["vel"])
            )
        self.gravity = 4 if self.gravity >= 2 else 1

        max_action = (
            0 if self.Q(state, 0)
            >= self.Q(state, 1) else 1
        )
        rand_action = int(npr.rand() < 0.4)
        new_action = rand_action if npr.rand() < self.epsilon else max_action

        r = self.last_reward
        s = self.last_state
        a = self.last_action
        sp = state
        ap = max_action
        gamma = self.gamma
        eta = self.eta
        # update w
        self.w[self.disc(s)][a] = (
            self.Q(s, a) - eta * (
                self.Q(s, a) - (r + gamma * self.Q(sp, ap))
            )
        )

        self.last_action = new_action
        self.last_state = state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters=1000, t_len=1):
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
    run_games(agent, hist)

    # Save history.
    np.save('hist', np.array(hist))
