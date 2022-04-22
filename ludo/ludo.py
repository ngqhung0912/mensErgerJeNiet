import numpy as np
import pygame
from pygame.locals import *
import time
import sys

SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 1024

CONVERT = ([4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [3, 6], [2, 6], [1, 6], [0, 6], [0, 5],
           [0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0], [5, 0],
           [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [7, 4], [8, 4], [9, 4], [10, 4], [10, 5],
           [10, 6], [9, 6], [8, 6], [7, 6], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10], [5, 10])

HOME = [([0, 9], [1, 9], [0, 10], [1, 10]),
        ([0, 0], [1, 0], [0, 1], [1, 1]),
        ([9, 0], [10, 0], [9, 1], [10, 1]),
        ([9, 9], [9, 10], [10, 9], [10, 10])]

TARGET = [([5, 9], [5, 8], [5, 7], [5, 6]),
          ([1, 5], [2, 5], [3, 5], [4, 5]),
          ([5, 1], [5, 2], [5, 3], [5, 4]),
          ([9, 5], [8, 5], [7, 5], [6, 5])]

DICE_POS = [(250, 774), (250, 250), (774, 250), (774, 774)]

YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (100, 100, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

COLORS = (YELLOW, RED, BLUE, GREEN)

START_OFFSET = [0, 10, 20, 30]


def rel2abs(rel_pos, offset):
    """
        returns the absolute position on the board (1..40) given the relative position and offset of a player
        """
    return (rel_pos - 1 + offset) % 40 + 1  # exceeding 40 wrap back to 1


class Dice:
    """
    defines a dice for rendering on the board
    """

    def __init__(self, screen):
        self.screen = screen
        self.width = 100
        self.border_width = 5
        self.diameter = 10
        self.eyes = (((0.5, 0.5),),
                     ((0.25, 0.25), (0.75, 0.75)),
                     ((0.25, 0.75), (0.50, 0.50), (0.75, 0.25)),
                     ((0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)),
                     ((0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75), (0.50, 0.50)),
                     ((0.25, 0.25), (0.25, 0.50), (0.25, 0.75), (0.75, 0.25), (0.75, 0.50), (0.75, 0.75))
                     )  # relative location of the eyes on the dice for the 6 sides

    def render(self, xpos, ypos, eyes):
        """
        draws the dice at the given centered position (xpos,ypos)
        """
        rect = (int(xpos - self.width / 2), int(ypos - self.width / 2), self.width, self.width)
        pygame.draw.rect(self.screen, WHITE, rect, 0)
        pygame.draw.rect(self.screen, BLACK, rect, self.border_width)
        for dx, dy in self.eyes[eyes - 1]:
            pos = (rect[0] + self.width * dx, rect[1] + self.width * dy)
            pygame.draw.circle(self.screen, BLACK, pos, self.diameter, 0)


class Renderer:
    """
    class that renders the gamestate using pygame
    """

    def __init__(self):
        # for rendering initialize pygame
        pygame.init()
        pygame.font.init()  # you have to call this at the start,
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.image = pygame.image.load(r'ludo/board.png')
        self.rect = self.image.get_rect()
        self.dice = Dice(self.screen)

    def render(self, obs, info):
        # check if user presses the quit in the GUI.
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        # draw the board and the dice
        self.screen.blit(self.image, self.rect)
        self.dice.render(*DICE_POS[info['player']], info['eyes'])

        # draw the pawns
        for id, positions in enumerate(obs):
            home_index = 0
            target_index = 0
            for position in positions:
                if 1 <= position <= 40:  # on the board
                    x, y = CONVERT[rel2abs(position, START_OFFSET[id]) - 1]

                elif position == 0:  # in the home square
                    x, y = HOME[id][home_index]
                    home_index += 1

                else:  # in the target area
                    x, y = TARGET[id][position - 41]
                    target_index += 1

                pygame.draw.circle(self.screen, COLORS[id], (38 + x * 95, 38 + y * 95), 25, 0)
                pygame.draw.circle(self.screen, BLACK, (38 + x * 95, 38 + y * 95), 25, 5)
                # pygame.draw.circle(self.screen, BLACK, (38 + x*95, 38 + y*95), 5, 0)

        pygame.display.flip()


class Game:
    def __init__(self, num_players=4, render=True):
        self.num_players = num_players
        self.start_pos = START_OFFSET  # fields in front of the players home
        self.players = [Player(self.start_pos[ii]) for ii in range(num_players)]
        self.rng = np.random.default_rng()
        self.render_valid = render
        self.eyes = ''
        self.current_player = 0
        self.previous_obs = None
        if self.render_valid:
            self.renderer = Renderer()

    def roll_dice(self):
        self.eyes = self.rng.integers(1, 7)
        return self.eyes

    def get_game_state(self):
        # returns the game state
        info = {'player': self.current_player,
                'eyes': self.eyes}
        obs = [player.obs() for player in self.players]
        reward = 0  # currently, always zero
        self.previous_obs = obs
        # compute if a player has won
        done = False
        for state in obs:
            if all(pos > 40 for pos in state):
                done = True

        return obs, reward, done, info

    def reset(self):
        # reset all players
        for player in self.players:
            player.reset()
        self.current_player = 0

        # return game state
        self.roll_dice()
        return self.get_game_state()

    def step(self, action):
        # compute new position of a pawn
        new_abs_pos, moved_pawn = self.players[self.current_player].step(action, self.eyes)

        # if a pawn has moved to a new position on the board check if another pawn is "geslagen"
        if 1 <= new_abs_pos <= 40:
            for id, player in enumerate(self.players):
                if id != self.current_player:
                    player.sla(new_abs_pos)

        # move to next player if turn of current player has ended
        if self.eyes != 6 or new_abs_pos == -1:  # if not a 6 was thrown or no move was possible: turn has ended
            self.current_player += 1

        # if last player had his turn move back to player 0    
        self.current_player = self.current_player % self.num_players

        # return the new state for the next player
        self.roll_dice()
        obs, reward, done, info = self.get_game_state()
        return obs, reward, done, info, moved_pawn

    def render(self):
        if self.render_valid:
            obs, _, _, info = self.get_game_state()
            self.renderer.render(obs, info)


class Pawn:
    """
    class to handle a pawn
    """

    def __init__(self, offset):
        self.position = 0  # field 0 is Home position
        self.offset = offset  # offset field for absolute position on board
        self.abs_position = self.calc_abs_position()  # absolute position on the board if pawn is on the board
        # otherwise 0

    def reset(self):
        self.position = 0

    def calc_abs_position(self):
        """
        returns the absolute position on the board (1..40)
        if pawn is not on the board returns 0
        """
        if self.position == 0 or self.position > 40:  # pawn not on board
            return self.position
        else:
            return rel2abs(self.position, self.offset)

    def step(self, eyes, blocked_positions):
        """
        move the pawn with eyes steps.
        blocked_positions are positions of pawns of the same player
        returns the absolute position on the board
        """
        if self.position == 0:
            if eyes == 6:
                new_position = 1  # place a pawn on the board
            else:
                return -1  # move is not possible

        elif self.position > 40:
            return -1  # cannot move pawn in target field

        else:
            new_position = self.position + eyes
            # count back at the end of target field
            if new_position > 44:
                new_position = 88 - new_position

        # check if there are other pawns of the same player
        if new_position in blocked_positions:
            return -1

        else:  # make the move
            self.position = new_position
            self.abs_position = self.calc_abs_position()
            return self.abs_position

    def sla(self, abs_position):
        if self.abs_position == abs_position:
            self.position = 0
            self.abs_position = self.calc_abs_position()


class Player:
    """
    class that defines a player
    """

    def __init__(self, offset):
        self.offset = offset  # first position on the board when a 6 is thrown
        self.pawns = [Pawn(self.offset) for _ in range(4)]  # 4 pawns per player

    def reset(self):
        for pawn in self.pawns:
            pawn.reset()

    def board_positions(self):
        return [pawn.calc_abs_position() for pawn in self.pawns]

    def obs(self):
        return [pawn.position for pawn in self.pawns]

    def step(self, action, eyes):
        moved_pawn_index = -1
        for i in np.argsort(action)[::-1]:  # sort action in reverse way: pawn with the highest value has priority
            new_abs_pos = self.pawns[i].step(eyes, self.obs())
            if new_abs_pos != -1:
                moved_pawn_index = i
                break
        # return
        return new_abs_pos, moved_pawn_index

    def sla(self, abs_position):
        for pawn in self.pawns:
            pawn.sla(abs_position)


def make(num_players=4, render=True):
    if not (1 <= num_players <= 4):
        raise ValueError('number of players should be in the range 2 - 4')
    return Game(num_players, render)


def random_player(obs, info):
    """
    returns a random action
    """
    return np.random.random_sample(size=4)


def eager_player(obs, info):
    """
    plays the pawn in order of their advance. The furthest pawn is player first
    """

    return obs[info['player']]


def main():
    print('testing')
    myplayers = [eager_player, random_player, random_player, random_player]
    # myplayers = [random_player]
    env = make(num_players=len(myplayers))
    obs, reward, done, info = env.reset()
    print(obs, done, info)
    env.render()
    while not done:
        action = myplayers[info['player']](obs, info)
        obs, reward, done, info, _ = env.step(action)
        print(obs, done, info)
        env.render()
        time.sleep(1)
    env.render()
    input('game finished, press any key')


if __name__ == '__main__':
    main()
