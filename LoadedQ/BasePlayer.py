import numpy as np
from collections import defaultdict


class BasePlayer:
    def __init__(self):
        self.index = None
        self.current_reward = 0
        self.reward_list = []
        self.dice = None
        self.previous_obs = None
        self.relative_position = None

    def load_model(self):
        pass

    def handle_reward(self, obs: list):
        pass

    def calculate_relative_position(self, obs: list):
        player_obs = []
        relative_position = []
        for player_elem in obs[self.index]:
            forward_pos = []
            backward_pos = []
            for i in range(4):  # loop over all other player's pawns
                if i == self.index:
                    continue
                if player_elem == 0 or player_elem > 40:
                    continue
                else:
                    relative_factor = i - self.index
                    for enemy_elem in obs[i]:  # loop over each pawn of other player.
                        if enemy_elem == 0 or enemy_elem > 40:
                            # ignore it if the pawn is in starting or in homebase.
                            continue
                        enemy_elem = enemy_elem + 10 * relative_factor  # calculate enemy's relative position
                        if enemy_elem > 40:
                            enemy_elem -= 40
                        distance = (enemy_elem - player_elem)  # calculate relative distance
                        # store two closest behind and two closest in front if it is in range of 12.
                        if 0 < distance < 12:
                            forward_pos.append(distance)
                        elif -12 < distance < 0:
                            backward_pos.append(distance)
            backward_pos.sort()
            forward_pos.sort()
            while len(backward_pos) > 2:
                backward_pos.pop(-1)
            while len(forward_pos) > 2:
                forward_pos.pop(0)
            while len(backward_pos) < 2:
                backward_pos.append(0)
            while len(forward_pos) < 2:
                forward_pos.insert(0, 0)
            # if there is nothing in range of 12, add 0 instead.
            each_relative_position = (backward_pos + forward_pos)
            relative_position.append(each_relative_position)
            player_obs.append(player_elem)
        relative_position.insert(0, player_obs)
        return relative_position

    def handle_move(self, obs: list, info: dict):
        pass

    def handle_nn_input(self, pos: list):

        for i in range(len(pos)):
            for j in range(len(pos[i])):
                if i == 0:
                    pos[i][j] = pos[i][j] / 44
                else:
                    pos[i][j] = pos[i][j] / 12
        pos = np.array(pos).reshape((20, 1))

        nn_input = np.append(pos, self.dice/6).reshape((21, 1))
        return nn_input

    def save_previous_obs(self, obs: list):
        self.previous_obs = obs
