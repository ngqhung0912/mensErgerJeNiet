import PlayerType


class Player:
    def __init__(self, player_index: int, player_type: PlayerType):
        self.current_reward = 0
        self.reward_list = []
        self.type = player_type
        self.player_index = player_index

    def player(self):
        pass

    def handle_game_state(self, rew: int, obs: list, done: bool, info: dict):
        self.current_reward = rew
        self.reward_list.append(rew)

