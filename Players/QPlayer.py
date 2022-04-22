from Players.Player import Player
import numpy as np
from model.Agent import Agent
from model.ModifiedTensorBoard import ModifiedTensorBoard as mtb
global Q_player


class QPlayer(Player):
    def __init__(self, model_name: str, epsilon: float, episodes: int):
        super(QPlayer, self).__init__()
        self.agent = Agent(model_name, discount_rate=0.7, learning_rate=0.001, episodes=episodes, training=False,
                           model_dir='models/ludo1.model',
                           tensorboard=mtb(log_dir="logs/train_continue2".
                                           format(0.01, 0.95, episodes)))
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.epsilon_decay = (self.epsilon - self.min_epsilon) / 20000
        self.info_array = ['epsilon = {}'.format(self.epsilon),
                           'learning rate = {}'.format(self.agent.learning_rate),
                           'discount rate = {}'.format(self.agent.discount_rate),
                           'loss function: false',
                           'neural network = 21 - 42 - 25 - 4']

    def handle_move(self, obs: list, info: dict) -> np.ndarray:
        self.index = info['player']
        self.dice = info['eyes']
        self.relative_position = Player.calculate_relative_position(self, obs)
        if np.random.random() > self.epsilon:
            nn_input = self.handle_nn_input(self.relative_position)
            move = self.agent.get_qs(nn_input).reshape((4,))
        else:
            move = np.random.random_sample(size=4)
        return move

    def handle_endgame(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)
        self.previous_obs = None

