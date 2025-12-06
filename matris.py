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

class Action(Enum):
    RIGHT = 0
    LEFT = 1
    DOWN = 2
    ROTATE = 3
    HARD_DROP = 4
    # NONE = 5

class Piece:
    def __init__(self, shape, color, position = None):
        self.shape = shape
        self.color = color
        self.position = position if position is not None else ((0,4) if len(shape) == 2 else (0, 3))

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.uint8)

    def fits_in_matrix(self, piece: Piece) -> bool:
        """
        Checks if tetromino fits on the board
        """
        posY, posX = piece.position
        for x in range(posX, posX + len(piece.shape)):
            for y in range(posY, posY + len(piece.shape)):
                if self.grid[y][x] is False and piece.shape[y - posY][x - posX]:  # outside matrix
                    return False

        return True




class Matris(object):
    def __init__(self, hard_drop_scores = False, actions_until_drop = 10):
        self.surface = None

        self.matrix = dict()
        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):
                self.matrix[(y,x)] = None

        self.heights = dict()
        for x in range(MATRIX_WIDTH):
            self.heights[x] = 0

        self.empty_matrix = dict()
        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):
                self.empty_matrix[(y, x)] = None
        """
        `self.matrix` is the current state of the tetris board, that is, it records which squares are
        currently occupied. It does not include the falling tetromino. The information relating to the
        falling tetromino is managed by `self.set_tetrominoes` instead. When the falling tetromino "dies",
        it will be placed in `self.matrix`.
        """

        self.hard_drop_scores = hard_drop_scores

        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.set_tetrominoes()
        self.tetromino_rotation = 0
        self.downwards_timer = 0
        self.base_downwards_speed = 0.4 # Move down every 400 ms

        self.movement_keys = {'left': 0, 'right': 0}
        self.movement_keys_speed = 0.05
        self.movement_keys_timer = (-self.movement_keys_speed)*2

        self.level = 1
        self.score = 0
        self.lines = 0

        self.combo = 1 # Combo will increase when you clear lines with several tetrominos in a row
        
        self.paused = False

        self.actions_until_drop = actions_until_drop
        self.actions_left = self.actions_until_drop

    def reset(self):
        self.matrix = dict()
        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):
                self.matrix[(y,x)] = None

        self.next_tetromino = self.select_next()
        self.set_tetrominoes()

        self.level = 1
        self.score = 0
        self.lines = 0
        self.combo = 1
        self.paused = False
        self.downwards_timer = 0
        self.tetromino_rotation = 0

        return self.state()

    def select_next(self):
        return random.choice(list_of_tetrominoes)

    def set_tetrominoes(self):
        """
        Sets information for the current and next tetrominos
        """
        self.current_tetromino = self.next_tetromino
        self.next_tetromino = self.select_next()
        self.surface_of_next_tetromino = self.construct_surface_of_next_tetromino()
        self.tetromino_position = (0,4) if len(self.current_tetromino.shape) == 2 else (0, 3)
        self.tetromino_rotation = 0
        self.tetromino_block = self.block(self.current_tetromino.color)
        self.shadow_block = self.block(self.current_tetromino.color, shadow=True)

    
    def hard_drop(self):
        """
        Instantly places tetrominos in the cells below
        """
        amount = 0
        while self.request_movement('down'):
            amount += 1
        if self.hard_drop_scores:
            self.score += 10*amount

        self.lock_tetromino()

    def step(self, action, timepassed=0.01):
        score = self.score
        lines = self.lines
        pre_reward = self.reward_metric(self.lines)
        try:
            self.perform_action(action)
            should_redraw = self.update(timepassed)
            gameover = False
            next_state = self.state()
        except GameOver:
            should_redraw = True
            gameover = True
            next_state = self.state(True)
        post_reward = self.reward_metric(self.lines)
        reward = -0.0001 + (post_reward - pre_reward)
        if gameover:
            reward = -100

        return next_state, float(reward), lines, should_redraw, gameover

    def aggregate_height(self):
        for x in range(MATRIX_WIDTH):
            height = 0
            for y in range(MATRIX_HEIGHT):
                if self.matrix[(y,x)] is not None:
                    height = MATRIX_HEIGHT - y
                    break
            self.heights[x] = height
        return sum(self.heights), self.heights

    def holes(self):
        holes = 0
        for x in range(MATRIX_WIDTH):
            for y in range(1, MATRIX_HEIGHT):
                if self.matrix[(y,x)] is None and self.matrix[(y-1,x)] is not None:
                    holes += 1
        return holes

    def bumpy(self):
        aggregate_height, heights = self.aggregate_height()
        bumps = 0
        for h in range(len(heights) - 1):
            bumps += abs(heights[h] - heights[h+1])
        return bumps, aggregate_height, heights

    def reward_metric(self, lines):
        holes = self.holes()
        bumps, aggregate_height, heights = self.bumpy()
        return -0.510066 * aggregate_height + 0.760666 * lines + -0.35663 * holes + -0.184483 * bumps

    def empty(self):
        return np.zeros((2, MATRIX_HEIGHT, MATRIX_WIDTH), dtype=np.float32)

    def state(self, empty=False):
        with_tetromino = self.blend(shape=self.current_tetromino.shape, position=self.tetromino_position, matrix=self.empty_matrix)
        state = self.empty()
        if not empty:
            try:
                for y, x in self.matrix:
                    value = self.matrix[(y,x)]
                    state[0][y][x] = 0 if value is None else 1
                for y, x in with_tetromino:
                    value = with_tetromino[(y,x)]
                    state[1][y][x] = 0 if value is None else 1
            except TypeError:
                pass

        return state

    def perform_action(self, action):
        if action == Action.LEFT:
            self.request_movement('left')
        elif action == Action.RIGHT:
            self.request_movement('right')
        elif action == Action.HARD_DROP:
            self.hard_drop()
        elif action == Action.ROTATE:
            self.request_rotation()
        elif action == Action.DOWN:
            if not self.request_movement('down'):
                self.lock_tetromino()
        self.actions_left -= 1
        if self.actions_left <= 0:
            self.actions_left = self.actions_until_drop
            if not self.request_movement('down'):
                self.lock_tetromino()

    def get_user_actions(self):
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
        return user_actions

    def update(self, timepassed):
        """
        Main game loop
        """
        self.needs_redraw = False

        if self.surface:
            pressed = lambda key: event.type == pygame.KEYDOWN and event.key == key
            unpressed = lambda key: event.type == pygame.KEYUP and event.key == key

            events = pygame.event.get(eventtype=[pygame.QUIT])
            #Controls pausing and quitting the game.
            for event in events:
                if pressed(pygame.K_p):
                    self.surface.fill((0,0,0))
                    self.needs_redraw = True
                    self.paused = not self.paused
                elif event.type == pygame.QUIT:
                    self.gameover(full_exit=True)
                elif pressed(pygame.K_ESCAPE):
                    self.gameover()

            if self.paused:
                return self.needs_redraw

            # for event in events:
            #     #Controls movement of the tetromino
            #     if pressed(pygame.K_SPACE):
            #         self.hard_drop()
            #     elif pressed(pygame.K_UP) or pressed(pygame.K_w):
            #         self.request_rotation()
            #     elif pressed(pygame.K_LEFT) or pressed(pygame.K_a):
            #         self.request_movement('left')
            #         self.movement_keys['left'] = 1
            #     elif pressed(pygame.K_RIGHT) or pressed(pygame.K_d):
            #         self.request_movement('right')
            #         self.movement_keys['right'] = 1
            #
            #     elif unpressed(pygame.K_LEFT) or unpressed(pygame.K_a):
            #         self.movement_keys['left'] = 0
            #         self.movement_keys_timer = (-self.movement_keys_speed)*2
            #     elif unpressed(pygame.K_RIGHT) or unpressed(pygame.K_d):
            #         self.movement_keys['right'] = 0
            #         self.movement_keys_timer = (-self.movement_keys_speed)*2


        # self.downwards_speed = self.base_downwards_speed ** (1 + self.level/10.)

        # self.downwards_timer += timepassed
        # downwards_speed = self.downwards_speed*0.10 if any([pygame.key.get_pressed()[pygame.K_DOWN],
        #                                                     pygame.key.get_pressed()[pygame.K_s]]) else self.downwards_speed
        # if self.downwards_timer > downwards_speed:
        #     if not self.request_movement('down'): #Places tetromino if it cannot move further down
        #         self.lock_tetromino()
        #
        #     self.downwards_timer %= downwards_speed

        # if any(self.movement_keys.values()):
        #     self.movement_keys_timer += timepassed
        # if self.movement_keys_timer > self.movement_keys_speed:
        #     self.request_movement('right' if self.movement_keys['right'] else 'left')
        #     self.movement_keys_timer %= self.movement_keys_speed
        
        return self.needs_redraw

    def surface_check(self):
        if self.surface is None:
            self.surface = screen.subsurface(Rect((MATRIS_OFFSET + BORDERWIDTH, MATRIS_OFFSET + BORDERWIDTH),
                                     (MATRIX_WIDTH * BLOCKSIZE, (MATRIX_HEIGHT - 2) * BLOCKSIZE)))

    def draw_surface(self):
        self.surface_check()
        """
        Draws the image of the current tetromino
        """
        with_tetromino = self.blend(matrix=self.place_shadow())

        for y in range(MATRIX_HEIGHT):
            for x in range(MATRIX_WIDTH):

                #                                       I hide the 2 first rows by drawing them outside of the surface
                block_location = Rect(x*BLOCKSIZE, (y*BLOCKSIZE - 2*BLOCKSIZE), BLOCKSIZE, BLOCKSIZE)
                if with_tetromino[(y,x)] is None:
                    self.surface.fill(BGCOLOR, block_location)
                else:
                    if with_tetromino[(y,x)][0] == 'shadow':
                        self.surface.fill(BGCOLOR, block_location)
                    
                    self.surface.blit(with_tetromino[(y,x)][1], block_location)
                    
    def gameover(self, full_exit=False):
        """
        Gameover occurs when a new tetromino does not fit after the old one has died, either
        after a "natural" drop or a hard drop by the player. That is why `self.lock_tetromino`
        is responsible for checking if it's game over.
        """

        write_score(self.score)
        
        if full_exit:
            exit()
        else:
            raise GameOver("Sucker!")

    def place_shadow(self):
        """
        Draws shadow of tetromino so player can see where it will be placed
        """
        posY, posX = self.tetromino_position
        while self.blend(position=(posY, posX)):
            posY += 1

        position = (posY-1, posX)

        return self.blend(position=position, shadow=True)

    def fits_in_matrix(self, shape, position):
        """
        Checks if tetromino fits on the board
        """
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                if self.matrix.get((y, x), False) is False and shape[y-posY][x-posX]: # outside matrix
                    return False

        return position
                    

    def request_rotation(self):
        """
        Checks if tetromino can rotate
        Returns the tetromino's rotation position if possible
        """
        rotation = (self.tetromino_rotation + 1) % 4
        shape = self.rotated(rotation)

        y, x = self.tetromino_position

        position = (self.fits_in_matrix(shape, (y, x)) or
                    self.fits_in_matrix(shape, (y, x+1)) or
                    self.fits_in_matrix(shape, (y, x-1)) or
                    self.fits_in_matrix(shape, (y, x+2)) or
                    self.fits_in_matrix(shape, (y, x-2)))
        # ^ That's how wall-kick is implemented

        if position and self.blend(shape, position):
            self.tetromino_rotation = rotation
            self.tetromino_position = position
            
            self.needs_redraw = True
            return self.tetromino_rotation
        else:
            return False
            
    def request_movement(self, direction):
        """
        Checks if teteromino can move in the given direction and returns its new position if movement is possible
        """
        posY, posX = self.tetromino_position
        if direction == 'left' and self.blend(position=(posY, posX-1)):
            self.tetromino_position = (posY, posX-1)
            self.needs_redraw = True
            return self.tetromino_position
        elif direction == 'right' and self.blend(position=(posY, posX+1)):
            self.tetromino_position = (posY, posX+1)
            self.needs_redraw = True
            return self.tetromino_position
        elif direction == 'up' and self.blend(position=(posY-1, posX)):
            self.needs_redraw = True
            self.tetromino_position = (posY-1, posX)
            return self.tetromino_position
        elif direction == 'down' and self.blend(position=(posY+1, posX)):
            self.needs_redraw = True
            self.tetromino_position = (posY+1, posX)
            return self.tetromino_position
        else:
            return False

    def rotated(self, rotation=None):
        """
        Rotates tetromino
        """
        if rotation is None:
            rotation = self.tetromino_rotation
        return rotate(self.current_tetromino.shape, rotation)

    def block(self, color, shadow=False):
        """
        Sets visual information for tetromino
        """
        colors = {'blue':   (105, 105, 255),
                  'yellow': (225, 242, 41),
                  'pink':   (242, 41, 195),
                  'green':  (22, 181, 64),
                  'red':    (204, 22, 22),
                  'orange': (245, 144, 12),
                  'cyan':   (10, 255, 226)}


        if shadow:
            end = [90] # end is the alpha value
        else:
            end = [] # Adding this to the end will not change the array, thus no alpha value

        border = Surface((BLOCKSIZE, BLOCKSIZE), pygame.SRCALPHA, 32)
        border.fill(list(map(lambda c: c*0.5, colors[color])) + end)

        borderwidth = 2

        box = Surface((BLOCKSIZE-borderwidth*2, BLOCKSIZE-borderwidth*2), pygame.SRCALPHA, 32)
        boxarr = pygame.PixelArray(box)
        for x in range(len(boxarr)):
            for y in range(len(boxarr)):
                boxarr[x][y] = tuple(list(map(lambda c: min(255, int(c*random.uniform(0.8, 1.2))), colors[color])) + end) 

        del boxarr # deleting boxarr or else the box surface will be 'locked' or something like that and won't blit.
        border.blit(box, Rect(borderwidth, borderwidth, 0, 0))


        return border

    def lock_tetromino(self):
        """
        This method is called whenever the falling tetromino "dies". `self.matrix` is updated,
        the lines are counted and cleared, and a new tetromino is chosen.
        """
        self.matrix = self.blend()

        lines_cleared = self.remove_lines()
        self.lines += lines_cleared

        if lines_cleared:
            self.score += 100 * (lines_cleared**2) * self.combo

        if self.lines >= self.level*10:
            self.level += 1

        self.combo = self.combo + 1 if lines_cleared else 1

        self.set_tetrominoes()

        if not self.blend():
            self.gameover()
            
        self.needs_redraw = True

    def remove_lines(self):
        """
        Removes lines from the board
        """
        lines = []
        for y in range(MATRIX_HEIGHT):
            #Checks if row if full, for each row
            line = (y, [])
            for x in range(MATRIX_WIDTH):
                if self.matrix[(y,x)]:
                    line[1].append(x)
            if len(line[1]) == MATRIX_WIDTH:
                lines.append(y)

        for line in sorted(lines):
            #Moves lines down one row
            for x in range(MATRIX_WIDTH):
                self.matrix[(line,x)] = None
            for y in range(0, line+1)[::-1]:
                for x in range(MATRIX_WIDTH):
                    self.matrix[(y,x)] = self.matrix.get((y-1,x), None)

        return len(lines)

    def blend(self, shape=None, position=None, matrix=None, shadow=False):
        """
        Does `shape` at `position` fit in `matrix`? If so, return a new copy of `matrix` where all
        the squares of `shape` have been placed in `matrix`. Otherwise, return False.
        
        This method is often used simply as a test, for example to see if an action by the player is valid.
        It is also used in `self.draw_surface` to paint the falling tetromino and its shadow on the screen.
        """
        if shape is None:
            shape = self.rotated()
        if position is None:
            position = self.tetromino_position

        copy = dict(self.matrix if matrix is None else matrix)
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                if (copy.get((y, x), False) is False and shape[y-posY][x-posX] # shape is outside the matrix
                    or # coordinate is occupied by something else which isn't a shadow
                    copy.get((y,x)) and shape[y-posY][x-posX] and copy[(y,x)][0] != 'shadow'):

                    return False # Blend failed; `shape` at `position` breaks the matrix

                elif shape[y-posY][x-posX]:
                    copy[(y,x)] = ('shadow', self.shadow_block) if shadow else ('block', self.tetromino_block)

        return copy

    def construct_surface_of_next_tetromino(self):
        """
        Draws the image of the next tetromino
        """
        shape = self.next_tetromino.shape
        surf = Surface((len(shape)*BLOCKSIZE, len(shape)*BLOCKSIZE), pygame.SRCALPHA, 32)

        for y in range(len(shape)):
            for x in range(len(shape)):
                if shape[y][x]:
                    surf.blit(self.block(self.next_tetromino.color), (x*BLOCKSIZE, y*BLOCKSIZE))
        return surf

class Game(object):
    def main(self, sc, matris):
        """
        Main loop for game
        Redraws scores and next tetromino each time the loop is passed through
        """
        self.clock = pygame.time.Clock()
        global screen
        screen = sc

        self.matris = matris
        
        screen.blit(construct_nightmare(screen.get_size()), (0,0))
        
        matris_border = Surface((MATRIX_WIDTH*BLOCKSIZE+BORDERWIDTH*2, VISIBLE_MATRIX_HEIGHT*BLOCKSIZE+BORDERWIDTH*2))
        matris_border.fill(BORDERCOLOR)
        screen.blit(matris_border, (MATRIS_OFFSET,MATRIS_OFFSET))
        
        self.redraw()
      

    def draw(self, framerate = 50):
        timepassed = self.clock.tick(framerate)
        if self.matris.update((timepassed / 1000.) if not self.matris.paused else 0):
            self.redraw()

    def step(self, timepassed):
        if self.matris.update(timepassed):
            self.redraw()

    def redraw(self):
        """
        Redraws the information panel and next termoino panel
        """
        if not self.matris.paused:
            self.blit_next_tetromino(self.matris.surface_of_next_tetromino)
            self.blit_info()

            self.matris.draw_surface()

        pygame.display.flip()


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
        scoresurf = renderpair("Score", self.matris.score)
        levelsurf = renderpair("Level", self.matris.level)
        linessurf = renderpair("Lines", self.matris.lines)
        combosurf = renderpair("Combo", "x{}".format(self.matris.combo))

        height = 20 + (levelsurf.get_rect().height + 
                       scoresurf.get_rect().height +
                       linessurf.get_rect().height + 
                       combosurf.get_rect().height )
        
        #Colours side panel
        area = Surface((width, height))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, width-BORDERWIDTH*2, height-BORDERWIDTH*2))
        
        #Draws side panel
        area.blit(levelsurf, (0,0))
        area.blit(scoresurf, (0, levelsurf.get_rect().height))
        area.blit(linessurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height))
        area.blit(combosurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height))

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
