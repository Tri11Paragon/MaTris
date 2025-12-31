#!/usr/bin/env python
import numpy as np
import pygame
from pygame import Rect, Surface
import random
import os
import MaTris.kezmenu.kezmenu as kezmenu

from MaTris.tetrominoes import list_of_tetrominoes
from MaTris.tetrominoes import rotate

from MaTris.scores import load_score, write_score
from enum import Enum

from MaTris.tetrominoes import tetrominoes_index


class GameOver(Exception):
    """Exception used for its control flow properties"""

def get_sound(filename):
    return pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), "resources", filename))

BGCOLOR = (15, 15, 20)
BORDERCOLOR = (140, 140, 140)

BLOCKSIZE = 30
BORDERWIDTH = 10

MATRIS_OFFSET = 20

MATRIX_WIDTH = 10
MATRIX_HEIGHT = 22

LEFT_MARGIN = 340

WIDTH = MATRIX_WIDTH*BLOCKSIZE + BORDERWIDTH*2 + MATRIS_OFFSET*2 + LEFT_MARGIN
HEIGHT = (MATRIX_HEIGHT-2)*BLOCKSIZE + BORDERWIDTH*2 + MATRIS_OFFSET*2

TRICKY_CENTERX = WIDTH-(WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2))/2

VISIBLE_MATRIX_HEIGHT = MATRIX_HEIGHT - 2

EMPTY_CELL = 0
SHADOW_CELL = 1
COLORED_CELL = 2

class Action(Enum):
    NONE = 0
    RIGHT = 1
    LEFT = 2
    DOWN = 3
    ROTATE = 4
    HARD_DROP = 5


class Piece:
    def __init__(self, shape, color, position = None, rotation=0):
        self.rotation = rotation
        self.shape = shape
        self.color = color
        self.position = position if position is not None else ((0,4) if len(shape) == 2 else (0, 3))

    def copy(self):
        return Piece(self.shape, self.color, tuple(self.position), self.rotation)

    def rotated(self, rotation=None):
        """
        Rotates tetromino
        """
        if rotation is None:
            rotation = self.rotation
        return Piece(rotate(self.shape, rotation), self.color, self.position, rotation)

    def move(self, y, x):
        return Piece(self.shape, self.color, (self.position[0] + y, self.position[1] + x), self.rotation)

    def request_rotation(self, grid):
        """
        Checks if tetromino can rotate
        Returns the tetromino's rotation position if possible
        """
        rotation = (self.rotation + 1) % 4
        shape = self.rotated(rotation)

        new_piece = (grid.fits_in_matrix(shape.move(0, 0)) or
                    grid.fits_in_matrix(shape.move(0, 1)) or
                    grid.fits_in_matrix(shape.move(0, -1)) or
                    grid.fits_in_matrix(shape.move(0, +2)) or
                    grid.fits_in_matrix(shape.move(0, -2)))
        # ^ That's how wall-kick is implemented

        if new_piece and grid.blend(new_piece) is not False:
            return Piece(self.shape, self.color, new_piece.position, rotation)
        else:
            return False
        
    def try_move_left(self, grid):
        if grid.blend(self.move(0, -1)) is not False:
            return self.move(0, -1)
        return False
    
    def try_move_right(self, grid):
        if grid.blend(self.move(0, 1)) is not False:
            return self.move(0, 1)
        return False
    
    def try_move_up(self, grid):
        if grid.blend(self.move(-1, 0)) is not False:
            return self.move(-1, 0)
        return False
    
    def try_move_down(self, grid):
        if grid.blend(self.move(+1, 0)) is not False:
            return self.move(1, 0)
        return False
    
    def __apply_move(self, grid, func):
        move = func(grid)
        if move is False:
            return False
        self.position = move.position
        return True
    
    def mut_move_left(self, grid):
        return self.__apply_move(grid, self.try_move_left)
    
    def mut_move_right(self, grid):
        return self.__apply_move(grid, self.try_move_right)
    
    def mut_move_down(self, grid):
        return self.__apply_move(grid, self.try_move_down)

    def should_lock(self, grid):
        piece1 = self.try_move_down(grid)
        return piece1 is False or piece1 is None

    def hard_drop(self, grid):
        """
        Instantly places tetrominos in the cells below
        """
        piece = self.copy()

        while piece.mut_move_down(grid):
            pass
        
        return piece

class Grid:
    def __init__(self, height, width):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.heights = np.zeros(width, dtype=np.int32)

    def from_state(self, state):
        grid = state[0]
        self.grid = grid.astype(np.uint8)
        self.heights = np.zeros(self.width, dtype=np.int32)
        return self

    def copy(self):
        grid = Grid(self.height, self.width)
        grid.grid = self.grid.copy()
        grid.heights = self.heights.copy()
        return grid

    def fits_in_matrix(self, piece: Piece):
        """
        Checks if tetromino fits on the board
        """
        posY, posX = piece.position
        for x in range(posX, posX + len(piece.shape)):
            for y in range(posY, posY + len(piece.shape)):
                if piece.shape[y - posY][x - posX] is None:
                    continue
                if x >= self.width or y >= self.height or x < 0:  # outside matrix
                    return False

        return piece

    def place_shadow(self, piece: Piece):
        """
        Draws shadow of tetromino so player can see where it will be placed
        """
        while self.blend(piece) is not False:
            piece = piece.move(1, 0)

        piece = piece.move(-1, 0)

        return self.blend(piece, shadow=True)

    def blend(self, piece: Piece, shadow=False, matrix=None):
        """
        Does `shape` at `position` fit in `matrix`? If so, return a new copy of `matrix` where all
        the squares of `shape` have been placed in `matrix`. Otherwise, return False.

        This method is often used simply as a test, for example to see if an action by the player is valid.
        It is also used in `self.draw_surface` to paint the falling tetromino and its shadow on the screen.
        """
        piece = piece.rotated()
        matrix = matrix if (matrix is not None and matrix is not False) else self.grid
        copy = matrix.copy()
        posY, posX = piece.position
        for x in range(posX, posX + len(piece.shape)):
            for y in range(posY, posY + len(piece.shape)):
                if piece.shape[y-posY][x-posX] is None:
                    continue
                if x >= self.width or y >= self.height or x < 0:  # shape is outside the matrix
                    # print(f"OUT OF BOUNDS ({y}, {x})")
                    return False  # Blend failed; `shape` at `position` breaks the matrix
                # coordinate is occupied by something else which isn't a shadow
                if copy[y][x] >= COLORED_CELL:
                    # print(f"COLLISION {copy[y][x]} @ ({y}, {x})")
                    return False
                copy[y][x] = SHADOW_CELL if shadow else COLORED_CELL

        return copy

    def clear(self):
        self.grid[:][:] = 0

    def place_tetromino(self, piece: Piece):
        """
        This method is called whenever the falling tetromino "dies". `self.matrix` is updated,
        the lines are counted and cleared, and a new tetromino is chosen.
        """
        # print()
        new_grid = self.blend(piece)

        if new_grid is False:
            return False
            # print("Why are we failing to lock?")
            # print(f"Trying to place {piece.shape} @ {piece.position}")
            # print(self.grid)
            # raise Exception("Why")

        self.grid = new_grid
        return True

    def remove_lines(self):
        """
        Removes lines from the board
        """
        lines = []
        for y in range(MATRIX_HEIGHT):
            #Checks if row if full, for each row
            line = (y, [])
            for x in range(MATRIX_WIDTH):
                if self.grid[y][x] >= COLORED_CELL:
                    line[1].append(x)
            if len(line[1]) == MATRIX_WIDTH:
                lines.append(y)

        for line in sorted(lines):
            #Moves lines down one row
            for x in range(MATRIX_WIDTH):
                self.grid[line][x] = EMPTY_CELL
            for y in range(0, line+1)[::-1]:
                for x in range(MATRIX_WIDTH):
                    self.grid[y][x] = self.grid[y-1][x] if y-1 >= 0 else EMPTY_CELL

        return len(lines)

    def aggregate_height(self):
        for x in range(MATRIX_WIDTH):
            height = 0
            for y in range(MATRIX_HEIGHT):
                if self.grid[y][x] >= COLORED_CELL:
                    height = MATRIX_HEIGHT - y
                    break
            self.heights[x] = height
        return sum(self.heights), self.heights

    def holes_old(self):
        holes = 0
        for x in range(MATRIX_WIDTH):
            for y in range(1, MATRIX_HEIGHT):
                if self.grid[y][x] < COLORED_CELL <= self.grid[y - 1][x]:
                    holes += 1
        return holes

    def holes(self):
        holes = 0
        for x in range(MATRIX_WIDTH):
            covered_squares = 0
            for y in reversed(range(0, MATRIX_HEIGHT)):
                # If the square is empty then we can increase the number of covered squares
                if self.grid[y][x] < COLORED_CELL:
                    covered_squares += 1
                # Otherwise, if it is full, we add the collected total of covered squares.
                # This allows us to account for multiple overhangs which we do not want.
                elif self.grid[y][x] >= COLORED_CELL:
                    holes += covered_squares
                    covered_squares = 0
        return holes

    def bumpy(self):
        aggregate_height, heights = self.aggregate_height()
        bumps = 0
        for h in range(len(heights) - 1):
            bumps += abs(heights[h] - heights[h+1])
        return bumps, aggregate_height, heights

    def row_filled(self):
        # compute the number of fully filled rows
        total = 0
        for y in range(1, MATRIX_HEIGHT):
            sub_total = 0
            for x in range(MATRIX_WIDTH):
                if self.grid[y][x] >= COLORED_CELL:
                    sub_total += 1
            total += sub_total / MATRIX_WIDTH
        return total / MATRIX_HEIGHT

    def brett_reward_metric(self):
        if self.grid is False or self.grid is None:
            return -np.inf
        return 0


    def reward_metric(self, lines):
        if self.grid is False or self.grid is None:
            return -np.inf
        holes = self.holes()
        bumps, aggregate_height, heights = self.bumpy()
        # return -0.510066 * aggregate_height + 0.760666 * lines + -0.35663 * holes + -0.184483 * bumps
        # from https://gyanigk.github.io/data/DMU_Final_Paper.pdf
        return 25 * (lines * lines) + 10 / (holes + 1) + 5 / (aggregate_height + 1) + 2 / (bumps + 1)


class Matris(object):
    def __init__(self, actions_until_drop = 10):
        self.grid = Grid(MATRIX_HEIGHT, MATRIX_WIDTH)
        """
        `self.matrix` is the current state of the tetris board, that is, it records which squares are
        currently occupied. It does not include the falling tetromino. The information relating to the
        falling tetromino is managed by `self.set_tetrominoes` instead. When the falling tetromino "dies",
        it will be placed in `self.matrix`.
        """

        self.next_tetromino = self.select_next()
        self.current_tetromino = self.select_next()

        self.lines = 0
        self.score = 0

        self.actions_until_drop = actions_until_drop
        self.actions_left = self.actions_until_drop

    def best_action_set(self, hard_drop_epsilon = 0.01):
        best_actions = None
        best_score = -np.inf
        piece = self.current_tetromino.copy()
        
        if not piece.try_move_down(self.grid):
            return [Action.DOWN]

        for rotation in range(4):
            # new_piece = piece.request_rotation(self.grid)
            new_piece = piece.copy()
            new_piece.rotation = rotation
            if self.grid.blend(new_piece) is False:
                continue
            piece = new_piece

            while piece.mut_move_left(self.grid):
                pass

            while True:
                grid = self.grid.copy()
                
                new_piece = piece.copy()
                while new_piece.mut_move_down(grid):
                    pass
                # new_piece = piece.hard_drop(grid)
                if grid.place_tetromino(new_piece):
                    lines = grid.remove_lines()
                    value = grid.reward_metric(lines)

                    if value > best_score:
                        best_actions = (rotation, random.random() <= hard_drop_epsilon, piece.copy())
                        best_score = value
                else:
                    print("For some reason we are failing to properly use this session")
                    
                if not piece.mut_move_right(self.grid):
                    break

        if best_actions is None:
            return [Action.NONE]

        actions = []
        old_y, old_x = self.current_tetromino.position
        new_y, new_x = best_actions[2].position

        diff_x = old_x - new_x

        for i in range(best_actions[0]):
            actions.append(Action.ROTATE)

        if diff_x > 0:
            for i in range(diff_x):
                actions.append(Action.LEFT)
        elif diff_x < 0:
            for i in range(-diff_x):
                actions.append(Action.RIGHT)

        if best_actions[1]:
            actions.append(Action.HARD_DROP)
        else:
            piece = best_actions[2].copy()
            while piece.mut_move_down(self.grid):
                actions.append(Action.DOWN)
            actions.append(Action.DOWN)

        return actions

    def reset(self):
        self.grid.clear()

        self.next_tetromino = self.select_next()
        self.set_tetrominoes()

        self.lines = 0
        self.score = 0

        return self.state(self.grid)

    def select_next(self):
        next_tet = random.choice(list_of_tetrominoes)
        return Piece(next_tet.shape, tetrominoes_index[next_tet.color])

    def set_tetrominoes(self):
        """
        Sets information for the current and next tetrominos
        """
        self.current_tetromino = self.next_tetromino
        self.next_tetromino = self.select_next()

        if self.grid.blend(self.current_tetromino) is False:
            self.gameover()

    def step(self, action, decay=True):
        if type(action) == int:
            action = Action(action)
        pre_reward = self.grid.reward_metric(self.lines)
        try:
            self.perform_action(action, decay=decay)
            game_over = False
            next_state = self.state(self.grid)
        except GameOver:
            game_over = True
            next_state = self.state(self.grid, True)
        post_reward = self.grid.reward_metric(self.lines)
        # reward = -0.000001 + (self.lines - score) ** 2
        reward = post_reward - pre_reward
        # if action in [Action.LEFT, Action.RIGHT, Action.ROTATE]:
        #     reward -= 0.0001 * abs(reward)
        if game_over:
            reward -= 100

        # print(reward)
        return next_state, float(reward), game_over

    def empty(self):
        return np.zeros((2, MATRIX_HEIGHT, MATRIX_WIDTH), dtype=np.float32)

    def state(self, grid, empty=False):
        state = self.empty()
        if not empty:
            state[0] = grid.grid.astype(np.float32)
            state[1] = grid.blend(self.current_tetromino, matrix=state[1])
        return state

    def scored(self, lines):
        self.lines += lines
        self.score += lines * 100
        if lines == 2:
            self.score += 100
        elif lines == 3:
            self.score += 200
        elif lines == 4:
            self.score += 400
            
    def place_tetromino(self, piece):
        self.grid.place_tetromino(piece)
        self.scored(self.grid.remove_lines())
        
    def place_current(self):
        self.place_tetromino(self.current_tetromino)
        self.set_tetrominoes()

    def get_best_column_action(self):
        piece = self.current_tetromino.copy()

        best_action = (0, 0)
        best_score = -np.inf

        for rotation in range(4):
            new_piece = piece.copy()
            new_piece.rotation = rotation

            if self.grid.blend(new_piece) is False:
                continue
            piece = new_piece

            while piece.mut_move_left(self.grid):
                pass

            for i in range(MATRIX_WIDTH):
                copy = piece.copy()
                grid = self.grid.copy()
                while copy.mut_move_down(grid):
                    pass

                grid.place_tetromino(copy)
                lines = grid.remove_lines()
                score = grid.reward_metric(lines)

                if score > best_score:
                    best_action = (rotation, i)
                    best_score = score

                piece.mut_move_right(grid)

        return best_action

    def get_columns_state(self):
        piece = self.current_tetromino.copy()

        states = {
            0: {},
            1: {},
            2: {},
            3: {}
        }

        for rotation in range(4):
            new_piece = piece.copy()
            new_piece.rotation = rotation

            if self.grid.blend(new_piece) is False:
                continue
            piece = new_piece

            while piece.mut_move_left(self.grid):
                pass

            for i in range(MATRIX_WIDTH):
                copy = piece.copy()
                grid = self.grid.copy()
                while copy.mut_move_down(grid):
                    pass

                grid.place_tetromino(copy)
                grid.remove_lines()

                states[rotation][i] = self.state(grid)
                piece.mut_move_right(grid)

        return states

    def place_in_column(self, rotation, column):
        cy, cx = self.current_tetromino.position

        diff = cx - column

        score = self.lines
        for i in range(rotation):
            self.perform_action(Action.ROTATE, False)
        try:
            if diff > 0:
                for i in range(diff):
                    self.perform_action(Action.LEFT, False)
            elif diff < 0:
                for i in range(-diff):
                    self.perform_action(Action.RIGHT, False)

            self.perform_action(Action.HARD_DROP, False)
            game_over = False
            next_state = self.state(self.grid)
        except GameOver:
            game_over = True
            next_state = self.state(self.grid, True)
        reward = (self.lines - score) ** 2
        if game_over:
            reward = -1000

        return next_state, float(reward), game_over


    def perform_action(self, action, decay=True):
        match action:
            case Action.LEFT:
                self.current_tetromino.mut_move_left(self.grid)
            case Action.RIGHT:
                self.current_tetromino.mut_move_right(self.grid)
            case Action.DOWN:
                if not self.current_tetromino.mut_move_down(self.grid):
                    self.place_current()
            case Action.HARD_DROP:
                new_piece = self.current_tetromino.hard_drop(self.grid)
                self.place_tetromino(new_piece)
                self.set_tetrominoes()
            case Action.ROTATE:
                new_piece = self.current_tetromino.request_rotation(self.grid)
                if new_piece is not False and new_piece is not None:
                    self.current_tetromino = new_piece
        if decay and action is not Action.DOWN:
            self.actions_left -= 1
            if self.actions_left <= 0:
                self.actions_left = self.actions_until_drop
                if not self.current_tetromino.mut_move_down(self.grid):
                    self.place_current()

    def gameover(self, full_exit=False):
        """
        Gameover occurs when a new tetromino does not fit after the old one has died, either
        after a "natural" drop or a hard drop by the player. That is why `self.lock_tetromino`
        is responsible for checking if it's game over.
        """
        
        if full_exit:
            exit()
        else:
            raise GameOver("Sucker!")

class Game(object):
    def main(self, sc, matris):
        """
        Main loop for game
        Redraws scores and next tetromino each time the loop is passed through
        """
        self.clock = pygame.time.Clock()
        self.surface = None
        global screen
        screen = sc

        self.matris = matris
        self.extra_keys = []
        
        screen.blit(construct_nightmare(screen.get_size()), (0,0))
        
        matris_border = Surface((MATRIX_WIDTH*BLOCKSIZE+BORDERWIDTH*2, VISIBLE_MATRIX_HEIGHT*BLOCKSIZE+BORDERWIDTH*2))
        matris_border.fill(BORDERCOLOR)
        screen.blit(matris_border, (MATRIS_OFFSET,MATRIS_OFFSET))
        
        self.redraw()

    def get_user_actions(self):
        self.extra_keys = []
        pygame.event.get(pygame.KEYDOWN)
        keyups = pygame.event.get(pygame.KEYUP)

        user_actions = []
        for event in keyups:
            if event.key == pygame.K_SPACE:
                user_actions.append(Action.HARD_DROP)
            elif event.key == pygame.K_UP or event.key == pygame.K_w:
                user_actions.append(Action.ROTATE)
            elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                user_actions.append(Action.LEFT)
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                user_actions.append(Action.RIGHT)
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                user_actions.append(Action.DOWN)
            elif event.key == pygame.K_ESCAPE:
                self.matris.gameover()
            else:
                self.extra_keys.append(event.key)
        return user_actions

    def is_key(self, key):
        return pygame.key.get_pressed()[key]

    def update(self):
        """
        Main game loop
        """
        if self.surface:
            pressed = lambda key: event.type == pygame.KEYDOWN and event.key == key
            unpressed = lambda key: event.type == pygame.KEYUP and event.key == key

            events = pygame.event.get(eventtype=[pygame.QUIT])
            # Controls pausing and quitting the game.
            for event in events:
                if event.type == pygame.QUIT:
                    self.matris.gameover(full_exit=True)

    def draw(self, framerate = 50):
        self.clock.tick(framerate)
        self.update()
        self.redraw()

    def step(self, timepassed):
        if self.matris.update(timepassed):
            self.redraw()

    def redraw(self):
        try:
            """
            Redraws the information panel and next termoino panel
            """
            surface_of_next_tetromino = self.construct_surface_of_next_tetromino()
            tetromino_block = self.block(self.matris.current_tetromino.color)
            shadow_block = self.block(self.matris.current_tetromino.color, shadow=True)
            self.blit_next_tetromino(surface_of_next_tetromino)
            self.blit_info()

            self.draw_surface(tetromino_block, shadow_block)
        except TypeError:
            pass

        pygame.display.flip()

    def surface_check(self):
        if self.surface is None:
            self.surface = screen.subsurface(Rect((MATRIS_OFFSET + BORDERWIDTH, MATRIS_OFFSET + BORDERWIDTH),
                                                  (MATRIX_WIDTH * BLOCKSIZE, (MATRIX_HEIGHT - 2) * BLOCKSIZE)))

    def draw_surface(self, tetromino_block, shadow_block):
        self.surface_check()
        """
        Draws the image of the current tetromino
        """
        # matrix=self.place_shadow()
        # matrix=self.grid.place_shadow(self.current_tetromino)

        with_tetromino = self.matris.grid.blend(self.matris.current_tetromino, matrix=self.matris.grid.place_shadow(self.matris.current_tetromino))

        if with_tetromino is False:
            with_tetromino = self.matris.grid.grid

        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):

                #                                       I hide the 2 first rows by drawing them outside of the surface
                block_location = Rect(x * BLOCKSIZE, (y * BLOCKSIZE - 2 * BLOCKSIZE), BLOCKSIZE, BLOCKSIZE)
                if with_tetromino[y][x] == EMPTY_CELL:
                    self.surface.fill(BGCOLOR, block_location)
                else:
                    if with_tetromino[y][x] == SHADOW_CELL:
                        self.surface.fill(BGCOLOR, block_location)
                        self.surface.blit(shadow_block, block_location)
                    else:
                        self.surface.blit(tetromino_block, block_location)

    def block(self, color, shadow=False):
        """
        Sets visual information for tetromino
        """
        colors = [(105, 105, 255),
                  (225, 242, 41),
                  (242, 41, 195),
                  (22, 181, 64),
                  (204, 22, 22),
                  (245, 144, 12),
                  (10, 255, 226)]

        if shadow:
            end = [90]  # end is the alpha value
        else:
            end = []  # Adding this to the end will not change the array, thus no alpha value

        border = Surface((BLOCKSIZE, BLOCKSIZE), pygame.SRCALPHA, 32)
        border.fill(list(map(lambda c: c * 0.5, colors[0])) + end)

        borderwidth = 2

        box = Surface((BLOCKSIZE - borderwidth * 2, BLOCKSIZE - borderwidth * 2), pygame.SRCALPHA, 32)
        boxarr = pygame.PixelArray(box)
        for x in range(len(boxarr)):
            for y in range(len(boxarr)):
                boxarr[x][y] = tuple(list(map(lambda c: min(255, int(c * random.uniform(0.8, 1.2))), colors[0])) + end)

        del boxarr  # deleting boxarr or else the box surface will be 'locked' or something like that and won't blit.
        border.blit(box, Rect(borderwidth, borderwidth, 0, 0))

        return border

    def construct_surface_of_next_tetromino(self):
        """
        Draws the image of the next tetromino
        """
        shape = self.matris.next_tetromino.shape
        surf = Surface((len(shape)*BLOCKSIZE, len(shape)*BLOCKSIZE), pygame.SRCALPHA, 32)

        for y in range(len(shape)):
            for x in range(len(shape)):
                if shape[y][x]:
                    surf.blit(self.block(self.matris.next_tetromino.color), (x*BLOCKSIZE, y*BLOCKSIZE))
        return surf

    def blit_info(self):
        """
        Draws information panel
        """
        textcolor = (255, 255, 255)
        font = pygame.font.Font(None, 30)
        width = (WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2)) - MATRIS_OFFSET*2

        def renderpair(text, val):
            text = font.render(text, True, textcolor)
            val = font.render(str(val), True, textcolor)

            surf = Surface((width, text.get_rect().height + BORDERWIDTH*2), pygame.SRCALPHA, 32)

            surf.blit(text, text.get_rect(top=BORDERWIDTH+10, left=BORDERWIDTH+10))
            surf.blit(val, val.get_rect(top=BORDERWIDTH+10, right=width-(BORDERWIDTH+10)))
            return surf
        
        #Resizes side panel to allow for all information to be display there.
        # scoresurf = renderpair("Score", self.matris.score)
        # levelsurf = renderpair("Level", self.matris.level)
        linessurf = renderpair("Lines", self.matris.lines)
        # combosurf = renderpair("Combo", "x{}".format(self.matris.combo))

        height = 20 + linessurf.get_rect().height
        
        #Colours side panel
        area = Surface((width, height))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, width-BORDERWIDTH*2, height-BORDERWIDTH*2))
        
        #Draws side panel
        # area.blit(levelsurf, (0,0))
        # area.blit(scoresurf, (0, levelsurf.get_rect().height))
        area.blit(linessurf, (0, 0))
        # area.blit(combosurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height))

        screen.blit(area, area.get_rect(bottom=HEIGHT-MATRIS_OFFSET, centerx=TRICKY_CENTERX))


    def blit_next_tetromino(self, tetromino_surf):
        """
        Draws the next tetromino in a box to the side of the board
        """
        area = Surface((BLOCKSIZE*5, BLOCKSIZE*5))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, BLOCKSIZE*5-BORDERWIDTH*2, BLOCKSIZE*5-BORDERWIDTH*2))

        areasize = area.get_size()[0]
        tetromino_surf_size = tetromino_surf.get_size()[0]
        # ^^ I'm assuming width and height are the same

        center = areasize/2 - tetromino_surf_size/2
        area.blit(tetromino_surf, (center, center))

        screen.blit(area, area.get_rect(top=MATRIS_OFFSET, centerx=TRICKY_CENTERX))

class Menu(object):
    """
    Creates main menu
    """
    running = True
    def main(self, screen):
        clock = pygame.time.Clock()
        menu = kezmenu.KezMenu(
            ['Play!', lambda: Game().main(screen)],
            ['Quit', lambda: setattr(self, 'running', False)],
        )
        menu.position = (50, 50)
        menu.enableEffect('enlarge-font-on-focus', font=None, size=60, enlarge_factor=1.2, enlarge_time=0.3)
        menu.color = (255,255,255)
        menu.focus_color = (40, 200, 40)
        
        nightmare = construct_nightmare(screen.get_size())
        highscoresurf = self.construct_highscoresurf() #Loads highscore onto menu

        timepassed = clock.tick(30) / 1000.

        while self.running:
            events = pygame.event.get()

            for event in events:
                if event.type == pygame.QUIT:
                    exit()

            menu.update(events, timepassed)

            timepassed = clock.tick(30) / 1000.

            if timepassed > 1: # A game has most likely been played 
                highscoresurf = self.construct_highscoresurf()

            screen.blit(nightmare, (0,0))
            screen.blit(highscoresurf, highscoresurf.get_rect(right=WIDTH-50, bottom=HEIGHT-50))
            menu.draw(screen)
            pygame.display.flip()

    def construct_highscoresurf(self):
        """
        Loads high score from file
        """
        font = pygame.font.Font(None, 50)
        highscore = load_score()
        text = "Highscore: {}".format(highscore)
        return font.render(text, True, (255,255,255))

def construct_nightmare(size):
    """
    Constructs background image
    """
    surf = Surface(size)

    boxsize = 8
    bordersize = 1
    vals = '1235' # only the lower values, for darker colors and greater fear
    arr = pygame.PixelArray(surf)
    for x in range(0, len(arr), boxsize):
        for y in range(0, len(arr[x]), boxsize):

            color = int(''.join([random.choice(vals) + random.choice(vals) for _ in range(3)]), 16)

            for LX in range(x, x+(boxsize - bordersize)):
                for LY in range(y, y+(boxsize - bordersize)):
                    if LX < len(arr) and LY < len(arr[x]):
                        arr[LX][LY] = color
    del arr
    return surf


if __name__ == '__main__':
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MaTris")
    Menu().main(screen)
