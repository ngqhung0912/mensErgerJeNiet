import numpy as np

class Player:
    def __init__(self):
        self.current_reward = 0
        self.reward_list = []

    def load_model(self):
        pass

    def handle_reward(self,obs: list, reward: int):
        pass

    def handle_move(self, obs: list, info: dict) -> np.ndarray:
        pass

