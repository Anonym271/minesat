import time
import random
from enum import Enum
from typing import Iterable
from pathlib import Path
import pygame


def get_random_coords(w, h, count):
    return [(i % w, i // w) for i in random.sample(range(0, w * h), count)]


class Resources:
    PATH = Path('./res')
    fields: dict = {}
    digits: dict = {}
    smileys: dict = {}

    @staticmethod
    def reload():
        fields =  {
            0: '0.png',
            1: '1.png',
            2: '2.png',
            3: '3.png',
            4: '4.png',
            5: '5.png',
            6: '6.png',
            7: '7.png',
            8: '8.png',
            'hidden': 'hidden.png',
            'marked': 'flag.png',
            'mine': 'mine.png',
            'mine_red': 'mine_red.png',
            'maybe': 'quest.png',
        }
        digits = {
            0: 'seg_0.png',
            1: 'seg_1.png',
            2: 'seg_2.png',
            3: 'seg_3.png',
            4: 'seg_4.png',
            5: 'seg_5.png',
            6: 'seg_6.png',
            7: 'seg_7.png',
            8: 'seg_8.png',
            9: 'seg_9.png',
            '-': 'seg_-.png',
            ' ': 'seg_none.png'
        }
        smileys = {
            'smile': 'smile.png',
            'tension': 'tension.png',
            'clear': 'clear.png',
            'dead': 'dead.png',
        }
        Resources.__reload_part(fields, Resources.fields)
        Resources.__reload_part(digits, Resources.digits)
        Resources.__reload_part(smileys, Resources.smileys)
    
    @staticmethod
    def __reload_part(source, target):
        for key, val in source.items():
            p = Resources.PATH.joinpath(val)
            target[key] = pygame.image.load(p)


class Field:
    SIZE = 16

    class States(Enum):
        HIDDEN = 0
        MARKED = 1
        MAYBE = 2
        VISIBLE = 3

    def __init__(self, is_mine: bool, hidden: bool = True):
        self.is_mine = is_mine
        self.state = Field.States.HIDDEN if hidden else Field.States.VISIBLE

    def __str__(self):
        if self.is_mine:
            return 'x'
        else: return ' '

    def __repr__(self):
        return self.__str__()


class BoardIterator:
    def __init__(self, board):
        self.board = board
        self.index = 0

    def __next__(self):
        x = self.index % self.board.width
        y = self.index // self.board.width
        if y >= self.board.height:
            raise StopIteration
        f = self.board[x, y]
        self.index += 1
        return (x, y), f


class Board:
    def __init__(self, width: int, height: int, mines: int | Iterable):
        self.width = width
        self.height = height
        if isinstance(mines, int):
            mines = get_random_coords(width, height, mines)
        self.mine_count = len(mines)
        mines = set(mines)
        self.fields = [
            [Field((x, y) in mines) for y in range(height)]
            for x in range(width)
        ]
        self.rect = pygame.Rect(0, 0, width * Field.SIZE, height * Field.SIZE)

    def __getitem__(self, key: int | tuple[int, int]):
        if isinstance(key, int):
            return self.fields[key]
        else:
            return self.fields[key[0]][key[1]]
    
    def __iter__(self):
        return BoardIterator(self)


class SevenSegmentDisplay(pygame.Rect):
    DIGIT_WIDTH = 13
    DIGIT_HEIGHT = 23

    def __init__(self, digits=3, value=0, position=(0, 0)):
        if digits <= 0:
            raise ValueError('Digit count must be > 0')
        pygame.Rect.__init__(self, position[0], position[1], SevenSegmentDisplay.DIGIT_WIDTH * digits, SevenSegmentDisplay.DIGIT_HEIGHT)
        self.surface = pygame.Surface(self.size)
        self.digits = digits
        self.value = value


    def render(self, value: int | None = None, leading_zero: bool = True):
        if value != None:
            self.value = value
        if self.value < 0:
            self.value = 0
        if self.value >= 10**self.digits:
            self.value = 10**self.digits - 1

        val = self.value
        for i in range(self.digits):
            d = val % 10
            self.surface.blit(Resources.digits[d], ((self.digits - i - 1) * SevenSegmentDisplay.DIGIT_WIDTH, 0))
            val //= 10
            if not leading_zero and val == 0:
                break
        
        return self.surface



class Button(pygame.Rect):
    class States(Enum):
        SMILE = 'smile'
        TENSION = 'tension'
        DEAD = 'dead'
        CLEAR = 'clear'
    
    SIZE = 24    
    
    def __init__(self, x, y):
        pygame.Rect.__init__(self, x, y, Button.SIZE, Button.SIZE)
        self.state = Button.States.SMILE

    def image(self):
        return Resources.smileys[self.state.value]



class MinesweeperGame:
    MARGIN = 5
    DEFAULT_WIDTH = 40
    DEFAULT_HEIGHT = 30
    
    class States(Enum):
        NONE = 0
        RUNNING = 1
        LOST = 2
        WON = 3
    

    def __init__(self, new_game=True):
        self.state = MinesweeperGame.States.NONE
        self.time_display = SevenSegmentDisplay(digits=3, position=(self.MARGIN, self.MARGIN))
        self.button = Button(self.time_display.right + self.MARGIN, self.MARGIN)
        self.mines_display = SevenSegmentDisplay(digits=3, position=(self.button.right + self.MARGIN, self.MARGIN))
        
        if new_game:
            self.new_game(size=(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT))
        else:   
            w = self.mines_display.right + self.MARGIN
            h = max(self.time_display.height, self.button.height) + 2 * self.MARGIN
            self.screen = pygame.display.set_mode((w, h))


    def new_game(self, board: Board = None, size: tuple[int, int] = None, mines: int = None):
        """Creates a new game either by using an existing board or by creating a new one.
        
        board: The board to use, if given. Define size instead to generate a new one.
        size: A tuple (x, y) that defines the size of the board. Leave this out if you want to use your own board.
        mines: The number of mines to use if a new board should be generated. Default is 15% of the fields."""
        if board == None:
            # Create new board
            if size == None:
                raise ValueError('Either board or size must be defined')
            x, y = size
            if x <= 0 or y <= 0:
                raise ValueError('Both field dimensions must be > 0')
            field_count = x * y
            if mines == None:
                mines = int(field_count * 0.15)
            elif mines >= field_count or mines <= 0:
                raise ValueError('Only 0 < mines < #fields is a valid mine count')
            board = Board(x, y, mines)
        
        self.board = board
        self.board.rect = self.board.rect.move(self.MARGIN, self.MARGIN * 2 + self.time_display.height)
        self.board_surface = pygame.Surface(self.board.rect.size)
        self.screen = pygame.display.set_mode(
            (self.board.rect.right + self.MARGIN, self.board.rect.bottom + self.MARGIN))

        self.end_time = None
        self.start_time = time.time()

        self.button.state = Button.States.SMILE
        self.state = MinesweeperGame.States.RUNNING
            

    def get_time(self):
        if self.end_time == None:
            return int(time.time() - self.start_time)
        else: 
            return int(self.end_time - self.start_time)


    def get_neighbor_coords(self, x: int | tuple[int, int], y: int | None = None):
        """Returns the coordinates of the 8 neighbor fields inside of the board.

        x: Either the x coordinate as int (then y is also required) or both at once as a tuple (x, y)
        y: The y coordinate as int. Leave it out if x is already a tuple containing both."""
        if isinstance(x, tuple):
            y = x[1]
            x = x[0]
        neighbors = [
            (x, y + 1),         # n
            (x, y - 1),         # s
            (x + 1, y),         # e
            (x - 1, y),         # w
            (x + 1, y + 1),     # ne
            (x - 1, y + 1),     # nw
            (x + 1, y - 1),     # se
            (x - 1, y - 1),     # sw
        ]
        return [
            pos for pos in neighbors 
            if pos[0] >= 0 and pos[0] < self.board.width and pos[1] >= 0 and pos[1] < self.board.height]
    

    def get_neighbor_fields(self,  x: int | tuple[int, int], y: int | None = None):
        """Returns the Field instances of the 8 neighbor fields inside of the board.

        x: either the x coordinate as int (then y is also required) or both at once as a tuple (x, y)
        y: the y coordinate as int. Leave it out if x is already a tuple containing both."""
        neighbors = self.get_neighbor_coords(x, y)
        return [self.board[crd] for crd in neighbors]


    def get_mine_count(self, x, y):
        count = 0
        for field in self.get_neighbor_fields(x, y):
            if field.is_mine:
                count += 1
        return count

    
    def get_marked_count(self):
        count = 0
        for pos, field in self.board:
            if field.state == Field.States.MARKED:
                count += 1
        return count
    

    def get_mines_left(self):
        total = self.board.mine_count
        marked = self.get_marked_count()
        return max(0, total - marked)


    def render(self):
        if self.state == self.States.NONE:
            self.screen.blit(self.time_display.render(0), self.time_display.topleft)
            self.screen.blit(self.mines_display.render(0), self.mines_display.topleft)
        else:
            for (x, y), f in self.board:
                s = f.state
                if s == Field.States.HIDDEN:
                    img = Resources.fields['hidden']
                elif s == Field.States.MARKED:
                    img = Resources.fields['marked']
                elif s == Field.States.MAYBE:
                    img = Resources.fields['maybe']
                elif s == Field.States.VISIBLE:
                    if f.is_mine:
                        if self.state == MinesweeperGame.States.LOST:
                            img = Resources.fields['mine_red']
                        else:
                            img = Resources.fields['mine']
                    else:
                        img = Resources.fields[self.get_mine_count(x, y)]
                if img != None: # Just to be sure
                    self.board_surface.blit(img, (x * Field.SIZE, y * Field.SIZE))

            self.screen.blit(self.board_surface, self.board.rect.topleft)
            self.screen.blit(self.time_display.render(self.get_time()), self.time_display.topleft)
            self.screen.blit(self.mines_display.render(self.get_mines_left()), self.mines_display.topleft)

        self.screen.blit(self.button.image(), self.button)
        pygame.display.flip()


    def show_field(self, x, y):
        field = self.board[x, y]
        field.state = Field.States.VISIBLE
        mines = self.get_mine_count(x, y)
        if mines == 0:
            for nx, ny in self.get_neighbor_coords(x, y):
                if self.board[nx, ny].state == Field.States.HIDDEN:
                    self.show_field(nx, ny)


    def show_all_mines(self):
        for pos, field in self.board:
            if field.is_mine:
                field.state = Field.States.VISIBLE


    def set_game_lost(self):
        self.state = MinesweeperGame.States.LOST
        self.button.state = Button.States.DEAD
        self.end_time = time.time()
        self.show_all_mines()


    def set_game_won(self):
        self.state = MinesweeperGame.States.WON
        self.button.state = Button.States.CLEAR
        self.end_time = time.time()
        self.show_all_mines()


    def check_game_won(self):
        # Def. won: all fields that do not contain a mine must be visible.
        for pos, field in self.board:
            if not field.is_mine and field.state != Field.States.VISIBLE:
                # A non-mine field that is not visible -> game not yet won
                return False
        self.set_game_won()
        return True


    def __mouse_to_board(self, mouse):
        if self.board.rect.collidepoint(mouse):
            x = mouse[0] - self.board.rect.left
            y = mouse[1] - self.board.rect.top
            x //= Field.SIZE
            y //= Field.SIZE
            return (x, y)
        return None
    

    def left_mouse_down(self, mouse):
        pass


    def left_mouse_up(self, mouse):
        if self.button.collidepoint(mouse[0], mouse[1]):
            self.new_game(size=(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT))
        elif self.state == MinesweeperGame.States.RUNNING:
            field_pos = self.__mouse_to_board(mouse)
            if field_pos != None:
                field = self.board[field_pos]
                x, y = field_pos
                if field.state == Field.States.HIDDEN:
                    if field.is_mine:
                        self.set_game_lost()
                    else:    
                        self.show_field(x, y)
                        self.check_game_won()
            

    def right_mouse_down(self, mouse):
        pass


    def right_mouse_up(self, mouse):
        if self.state == MinesweeperGame.States.RUNNING:
            field_pos = self.__mouse_to_board(mouse)
            if field_pos != None:
                field = self.board[field_pos]
                if field.state == Field.States.HIDDEN:
                    field.state = Field.States.MARKED
                elif field.state == Field.States.MARKED:
                    field.state = Field.States.MAYBE
                elif field.state == Field.States.MAYBE:
                    field.state = Field.States.HIDDEN


def main(width, height, mines=None):
    pygame.init()
    Resources.reload()
    MinesweeperGame.DEFAULT_WIDTH = width
    MinesweeperGame.DEFAULT_HEIGHT = height
    
    game = MinesweeperGame()

    shouldStop = False
    while not shouldStop:
        game.render()
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                shouldStop = True
            # Mouse events
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    game.left_mouse_down(mouse_pos)
                elif event.button == 3:
                    game.right_mouse_down(mouse_pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    game.left_mouse_up(mouse_pos)
                elif event.button == 3:
                    game.right_mouse_up(mouse_pos)


if __name__ == '__main__':
    # from argparse import ArgumentParser
    # args = ArgumentParser()
    # args.add_argument('width')
    # args.add_argument('height')
    # args.add_argument('mines')
    # args.parse_args()
    main(40, 30)