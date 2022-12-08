from Players.Player import Player
import numpy as np
from model.Agent import Agent
from model.ModifiedTensorBoard import ModifiedTensorBoard as mtb


class LoadedQPlayer(Player):
    """
    This class implements the base player class, and, contrary to the QPlayer.py - does not implement epsilon-greedy
    algorithm.
    """
    def __init__(self, model_name: str, episodes: int):
        """
        @param model_name: Name of the model.
        @param episodes:
        """
        super(LoadedQPlayer, self).__init__()
        self.agent = Agent(model_name, discount_rate=0.95, learning_rate=0.01, training=False,
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
            'loss function: Categorical_CE',
            'neural network = 21 - 16x4 - 4']

    def handle_move(self, obs: list, info: dict) -> np.ndarray:
        """
        Handle the move when it's this player's turn.
        @param obs: Current state returned by the game engine.
        @param info: Dictionary returned by the game engine, contains other players' states and the dice.
        """
        self.index = info['player']
        self.dice = info['eyes']
        self.relative_position = Player.calculate_relative_position(self, obs)
        nn_input = self.handle_nn_input(self.relative_position)
        move = self.agent.get_qs(nn_input).reshape((4,))
        return move

    def handle_endgame(self):
        """
        Handle when the game ends.
        """
        self.previous_obs = None
