from Players.Player import Player
import numpy as np

class StrategyPlayer(Player):
    def __init__(self):
        super(StrategyPlayer, self).__init__()

    def handle_move(self, obs: list, info: dict) -> np.ndarray:
        return np.ndarray([0, 0, 1, 0])
