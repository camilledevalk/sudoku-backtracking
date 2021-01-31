from time import time

import numpy as np
import pyximport;

pyximport.install(setup_args={'include_dirs': np.get_include()})
import subprocess
subprocess.call(["cython", "-a", "functions.pyx"])

import functions
from functions import get_board_to_play

BOARD_CONSTANT = 3
PRINT_INTERMEDIATE = False

def pretty_print(board, original_board):
    default_color = '\033[0;0m'
    special_color = '\033[96m'
    def print_row_of_dashes():
        for _ in range(BOARD_CONSTANT ** 2 + BOARD_CONSTANT + 1):
            print('-', end=' ')
    print('\n', end=' ')
    for i, row in enumerate(board):
        if i%BOARD_CONSTANT==0:
            print_row_of_dashes()
            print('\n', end=' ')
        for j, entry in enumerate(row):
            color = default_color
            if j%BOARD_CONSTANT==0:  # TODO write this more dynamic such that weird boxes also work
                # Vertical lines between the squares
                print(f'{default_color}|', end=' ')
            if entry != 0:  # If there is a number in the sudoku board
                # Check if it also appears in the global
                if entry != original_board[i,j]:
                    color = special_color
                output = f'{color}{entry}'
            else:  # Print nothing for the zeros
                output = ' '
            print(output, end=' ')
        print(f'{default_color}|', end=' ')

        print('\n', end=' ')
    print_row_of_dashes()
    print('')

board_to_play = get_board_to_play()
pretty_print(board_to_play, original_board=board_to_play)
tic = time()
solved = functions.run_backtrack()
tac = time()
print(f'Done in {tac-tic:1.5f}s')
pretty_print(solved, original_board=board_to_play)