from Players.Player import Player
import numpy as np
from model.Agent import Agent
from model.ModifiedTensorBoard import ModifiedTensorBoard as mtb

"""
This class implements the base player class, and, contrary to the QPlayer.py - does not implements epsilon-greedy 
algorithm.
"""


class LoadedQPlayer(Player):
    def __init__(self, model_name: str, episodes: int):
        super(LoadedQPlayer, self).__init__()
        self.agent = Agent(model_name, discount_rate=0.95, learning_rate=0.01, episodes=episodes, training=False,
                           tensorboard=mtb(log_dir="logs/testrun3".
                                           format(0.01, 0.95, episodes)),
                           model_dir='models/ludotrain_run.model')
        self.killed = 0
        self.being_killed = 0
        self.previous_pos = [0, 0, 0, 0]
        self.kicked = False
        self.previous_action = None
        self.info_array = [
                           'learning rate = {}'.format(self.agent.learning_rate),
                           'discount rate = {}'.format(self.agent.discount_rate),
                           'loss function: MAE',
                           'neural network = 21 - 16x4 - 4']

    def handle_move(self, obs: list, info: dict) -> np.ndarray:
        self.index = info['player']
        self.dice = info['eyes']
        self.relative_position = Player.calculate_relative_position(self, obs)
        nn_input = self.handle_nn_input(self.relative_position)
        move = self.agent.get_qs(nn_input).reshape((4,))
        return move

    def handle_endgame(self):
        self.previous_obs = None


