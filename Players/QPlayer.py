from Players.Player import Player
import numpy as np
from model.Agent import Agent

global Q_player


class QPlayer(Player):
    def __init__(self, model_name: str, epsilon: float, episodes: int):
        super(QPlayer, self).__init__()
        self.agent = Agent(model_name, discount_rate=0.95, learning_rate=0.01, episodes=episodes)
        self.epsilon = epsilon
        self.min_epsilon = 0.01
        self.epsilon_decay = (self.epsilon - self.min_epsilon) / 10000
        self.killed = 0
        self.being_killed = 0
        self.previous_pos = [0, 0, 0, 0]
        self.kicked = False
        self.info_array = ['epsilon = {}'.format(self.epsilon),
                           'learning rate = {}'.format(self.agent.learning_rate),
                           'discount rate = {}'.format(self.agent.discount_rate),
                           'loss function: false',
                           'neural network = 21 - 42 - 25 - 4']
        self.previous_action = None

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

    def update_memory(self, action, reward, done):
        if self.previous_obs is None or len(self.previous_obs) == 0:
            return
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

    def handle_reward(self, obs: list):
        reward = 0

        if self.index is None or self.previous_obs is None or len(self.previous_obs) == 0:
            return 0
        self.relative_position = Player.calculate_relative_position(self, obs)
        current_obs = obs[self.index]
        try:
            if max(filter(lambda x: x < 40, current_obs)) > max(filter(lambda x: x < 40, self.previous_pos)):
                # reward to moving closer to homebase                # reward to moving closer to homebase
                reward += 0.1
        except ValueError:
            pass

        for i in range(len(self.previous_pos)):
            if self.previous_pos[i] == 0 and current_obs[i] != 0:  # reward for releasing a piece
                reward += 0.25

        previous_relative_position = Player.calculate_relative_position(self, self.previous_obs)
        previous_relative_position.pop(0)
        previous_relative_position = np.array(previous_relative_position).reshape((16, 1))

        current_relative_position = self.relative_position.copy()
        current_relative_position.pop(0)
        current_relative_position = np.array(current_relative_position).reshape((16, 1))
        for i in range(previous_relative_position.shape[0]):
            if 0 > previous_relative_position[i] > -6 > current_relative_position[i] != 0:
                reward += 0.2

        if self.kicked:
            reward -= 0.5
            self.being_killed += 1
            self.kicked = False

        for i in range(len(obs)):  # calculate if a piece is knocked
            if i == self.index:
                continue
            else:
                for j in range(len(obs[i])):
                    if self.previous_obs[i][j] > 0 and obs[i][j] == 0:
                        reward += 1  # knocked a piece
                        self.killed += 1
        return reward

    def handle_endgame(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay
            self.epsilon = max(self.min_epsilon, self.epsilon)
        self.previous_obs = None

    def update_tensorboard_stats(self, episode_rewards_list: list, win_rate: float, aggregate_stats,
                                 time):
        average_reward = sum(episode_rewards_list[-aggregate_stats:]) / len(
            episode_rewards_list[-aggregate_stats:])
        min_reward = min(episode_rewards_list[-aggregate_stats:])
        max_reward = max(episode_rewards_list[-aggregate_stats:])
        self.agent.tensorboard.update_stats(reward_avg=average_reward,
                                            reward_min=min_reward,
                                            reward_max=max_reward,
                                            epsilon=self.epsilon,
                                            win_rate=win_rate,
                                            time=time,
                                            avg_being_knocked=self.being_killed / aggregate_stats,
                                            avg_killed=self.killed / aggregate_stats)
        self.killed = 0
        self.being_killed = 0

    def save_previous_pos(self, pos: list):
        self.previous_pos = pos

    def inform_kicked(self):
        self.kicked = True

    def save_action(self, action: list):
        self.previous_action = action

    def get_action(self):
        return self.previous_action
