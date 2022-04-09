from Players.Player import Player
import numpy as np
import os
from model.Agent import Agent


global Q_player


class QPlayer(Player):
    def __init__(self, model_name: str, epsilon: float, episodes: int):
        super(QPlayer, self).__init__()
        self.previous_obs = None
        self.relative_position = None
        self.agent = Agent(model_name, discount_rate=0.8, learning_rate=0.005, episodes=episodes)
        self.epsilon = epsilon
        self.min_epsilon = 0.001
        self.epsilon_decay = 0.998


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

    def save_previous_obs(self, obs: list):
        self.previous_obs = obs

    def update_memory(self, action, reward, done):

        prev_relative_pos = self.calculate_relative_position(self.previous_obs)
        prev_nn_input = self.handle_nn_input(prev_relative_pos)
        current_nn_input = self.handle_nn_input(self.relative_position)
        self.agent.update_replay_memory([
            prev_nn_input,
            action,
            reward,
            current_nn_input,
            done
        ])
        return

    def handle_reward(self, obs: list, current_player: int):
        reward = 0
        if self.index is None:
            return 0
        self.relative_position = Player.calculate_relative_position(self, obs)
        previous_relative_position = Player.calculate_relative_position(self, self.previous_obs)
        current_obs = obs[self.index]
        last_obs = self.previous_obs[self.index]
        if max(current_obs) > max(last_obs):  # reward to moving closer to homebase
            reward += 0.1*10

        for i in range(len(last_obs)):
            if last_obs[i] == current_obs[i]:
                continue
            elif last_obs[i] != 0 and current_obs[i] == 0:  # punish for being knocked
                reward -= 0.25*10
            elif last_obs[i] == 0 and current_obs[i] != 0:  # reward for releasing a piece
                reward += 0.25*10

        previous_relative_position.pop(0)
        previous_relative_position = np.array(previous_relative_position).reshape((16, 1))
        current_relative_position = self.relative_position.copy()

        current_relative_position.pop(0)
        current_relative_position = np.array(current_relative_position).reshape((16, 1))
        for i in range(previous_relative_position.shape[0]):
            if previous_relative_position[i] > 0 and current_relative_position[i] == 0:  # kicked a pawn
                reward += 0.15*10
            elif previous_relative_position[i] < 0 and \
                    previous_relative_position[i] - current_relative_position[i] < -6 \
                    and current_relative_position[i] != 0:
                reward += 0.25*10
        return reward

    def handle_endgame(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)



# def player(obs, info):
#     """
#     defines a random player: returns a random action as a (4,) numpy array
# regardless the game state
#     """
#     # here goes your code
#     # do not load your model here but use the main() function icm with a global variable
#     return Q_player.handle_move(obs, info)


# any other code that should run during import define in function main()
# def main():
#     # do all required initialisation here
#     # use relative paths for access to stored files that you require
#     # use global variables to make sure the player() function has access
#     global Q_player
#     model
#     Q_player = QPlayer(training_player = QPlayer(model_name, epsilon=1)
# )
#     pass
#
#
# main()
