from random import random

from tensorforce.environments import Environment
import numpy as np
import random

BOARD_HEIGHT = 6
BOARD_WIDTH = 6
BOARD_SIZE = BOARD_HEIGHT * BOARD_WIDTH
PLANE_SIZE = 8


def init_board():
    hidden_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH))

    # Populate the plane's position
    # First figure out the plane's orientation
    #   0: heading right
    #   1: heading up
    #   2: heading left
    #   3: heading down

    plane_orientation = random.randint(0, 3)

    # Figrue out plane core's position as the '*' below
    #        |         | |
    #       -*-        |-*-
    #        |         | |
    #       ---
    if plane_orientation == 0:
        plane_core_row = random.randint(1, BOARD_HEIGHT - 2)
        plane_core_column = random.randint(2, BOARD_WIDTH - 2)
        # Populate the tail
        hidden_board[plane_core_row][plane_core_column - 2] = 1
        hidden_board[plane_core_row - 1][plane_core_column - 2] = 1
        hidden_board[plane_core_row + 1][plane_core_column - 2] = 1
    elif plane_orientation == 1:
        plane_core_row = random.randint(1, BOARD_HEIGHT - 3)
        plane_core_column = random.randint(1, BOARD_WIDTH - 3)
        # Populate the tail
        hidden_board[plane_core_row + 2][plane_core_column] = 1
        hidden_board[plane_core_row + 2][plane_core_column + 1] = 1
        hidden_board[plane_core_row + 2][plane_core_column - 1] = 1
    elif plane_orientation == 2:
        plane_core_row = random.randint(1, BOARD_HEIGHT - 2)
        plane_core_column = random.randint(1, BOARD_WIDTH - 3)
        # Populate the tail
        hidden_board[plane_core_row][plane_core_column + 2] = 1
        hidden_board[plane_core_row - 1][plane_core_column + 2] = 1
        hidden_board[plane_core_row + 1][plane_core_column + 2] = 1
    elif plane_orientation == 3:
        plane_core_row = random.randint(2, BOARD_HEIGHT - 2)
        plane_core_column = random.randint(1, BOARD_WIDTH - 2)
        # Populate the tail
        hidden_board[plane_core_row - 2][plane_core_column] = 1
        hidden_board[plane_core_row - 2][plane_core_column + 1] = 1
        hidden_board[plane_core_row - 2][plane_core_column - 1] = 1

    # Populate the cross
    hidden_board[plane_core_row][plane_core_column] = 1
    hidden_board[plane_core_row + 1][plane_core_column] = 1
    hidden_board[plane_core_row - 1][plane_core_column] = 1
    hidden_board[plane_core_row][plane_core_column + 1] = 1
    hidden_board[plane_core_row][plane_core_column - 1] = 1

    return hidden_board.reshape(BOARD_SIZE, )


class PlaneStrike(Environment):
    def __init__(self):
        pass

    def __str__(self):
        return 'MinimalTest'

    def close(self):
        pass

    def reset(self):
        self.state = np.zeros(N)
        self.hidden_state = init_board()
        self.count = 0
        return self.state

    def execute(self, action):
        if self.state[action] == 1 and self.state[action] == -1:
            reward = -1
        else:
            if self.hidden_state[action] == 1:
                self.state[action] = 1
                self.count = self.count + 1
                reward = 1
            else:
                self.state[action] = -1
                reward = -1
        terminal = (self.count == PLANE_SIZE)
        return self.state, reward, terminal

    @property
    def states(self):
        return dict(shape=(N,), type='float')

    @property
    def actions(self):
        return dict(continuous=False, num_actions=N)
