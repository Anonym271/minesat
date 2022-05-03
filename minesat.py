from pathlib import Path
import os
from Layout import FieldBuilder, Field
from BooleanParser import BooleanParser, SyntaxTree
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame



TEX_SIZE = 16


class Resources:
    PATH = Path('./res')
    fields: dict = {}

    @staticmethod
    def reload():
        fields =  {
            '0': '0.png',
            '1': '1.png',
            '2': '2.png',
            '3': '3.png',
            '4': '4.png',
            '5': '5.png',
            '6': '6.png',
            '7': '7.png',
            '8': '8.png',
            'hidden': 'hidden.png',
            'marked': 'flag.png',
            'mine': 'mine.png',
            'mine_red': 'mine_red.png',
            'maybe': 'quest.png',
        }
        Resources.__reload_part(fields, Resources.fields)
    
    @staticmethod
    def __reload_part(source, target):
        for key, val in source.items():
            p = Resources.PATH.joinpath(val)
            target[key] = pygame.image.load(p)



def render_field(field: Field) -> pygame.Surface:
    surf = pygame.Surface((field.width * TEX_SIZE, field.height * TEX_SIZE))
    for x in range(field.width):
        for y in range(field.height):
            c = field[x, y]
            if c.isdigit():
                img = Resources.fields[c]
            elif c.isalpha(): # A variable
                img = Resources.fields['hidden']
            elif c == '*':
                img = Resources.fields['marked']
            else:
                img = Resources.fields['0']
            surf.blit(img, (x * TEX_SIZE, y * TEX_SIZE))
    return surf



def main_render(formula: SyntaxTree):
    pygame.init()
    field = FieldBuilder(formula).build()
    field_surf = render_field(field)
    print(field_surf.get_size())
    win = pygame.display.set_mode(field_surf.get_size())
    win.blit(field_surf, (0, 0))

    # img = pygame.image.load('res/0.png')
    pygame.display.flip()
    
    should_stop = False
    while not should_stop:
        # win.blit(img, (50, 50))
        for event in pygame.event.get():
            t = event.type
            if t == pygame.QUIT or t == pygame.KEYDOWN or t == pygame.MOUSEBUTTONDOWN:
                should_stop = True



def main_save(formula: SyntaxTree, filename):
    field = FieldBuilder(formula).build()
    field_surf = render_field(field)
    pygame.image.save(field_surf, filename)


Resources.reload()

if __name__ == '__main__':
             

    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument('-s', '--save', action='store_true', help='Only save the image')
    args.add_argument('-d', '--dpioff', action='store_true', help='Disable DPI scaling on windows')
    args.add_argument('formula', help='The formula to render')
    args.add_argument('filename', nargs='?', default='output.png', help='Destination file name (for save mode)')
    args = args.parse_args()

    if args.dpioff:
        import ctypes
        err_code = ctypes.windll.shcore.SetProcessDpiAwareness(2)   
        if err_code != 0:
            print(f'Failed to set DPI awareness (Error code {err_code})')

    formula = BooleanParser(args.formula).parse()
    if args.save:
        main_save(formula, args.filename)
    else:
        main_render(formula)