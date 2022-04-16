from Players.Player import Player
import numpy as np


class StrategyPlayer(Player):
    def __init__(self):
        super(StrategyPlayer, self).__init__()

    def handle_move(self, obs: list, info: dict) -> np.ndarray:
        self.index = info['player']
        self.dice = info['eyes']

        move = np.random.random_sample(size=4)

        self.relative_position = Player.calculate_relative_position(self, obs)
        future_obs = obs

        current_pawn_position = self.relative_position.copy()
        current_pawn_position = current_pawn_position.pop(0)
        enemy_location = np.array(self.relative_position[1:5]).reshape((16, 1))

        closest_pawn_front = np.argmax(enemy_location)
        closest_pawn_behind = np.argmin(enemy_location)

        min_pawn = np.argmin(current_pawn_position)
        max_pawn = np.argmax(current_pawn_position)

        if current_pawn_position[max_pawn] == 0:
            return move

        if enemy_location[closest_pawn_front] == self.dice:
            i = 0
            if 0 <= closest_pawn_front < 4:
                i = 1
            elif 4 <= closest_pawn_front <= 8:
                i = 1
            elif 8 <= closest_pawn_front < 12:
                i = 1
            elif 12 <= closest_pawn_front < 16:
                i = 1
            move[i] = 1
            return move

        if -6 < enemy_location[closest_pawn_behind] < 0:
            i = 0
            if 0 <= closest_pawn_behind < 4:
                i = 1
            elif 4 <= closest_pawn_behind <= 8:
                i = 1
            elif 8 <= closest_pawn_behind < 12:
                i = 1
            elif 12 <= closest_pawn_behind < 16:
                i = 1

            move[i] = 1
            return move

        if self.dice == 6:
            if current_pawn_position[max_pawn] > 30:
                move[max_pawn] = 1
                return move
            if current_pawn_position[min_pawn] == 0:
                move[min_pawn] = 1
                return move
        for i in range(len(current_pawn_position)):
            if current_pawn_position[i] == 1:
                move[i] = 1
                return move

        move[max_pawn] = 1
        return move

