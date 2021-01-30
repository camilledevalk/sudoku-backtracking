from time import time

import numpy as np
import pyximport;

pyximport.install(setup_args={'include_dirs': np.get_include()})
import subprocess
subprocess.call(["cython", "-a", "functions.pyx"])

import functions

EMPTY_BOARD = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]],
    dtype=np.int8
)

# Easy board
easy_board = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]],
    dtype=np.int8
)

# Hard board
hard_board = np.array([
    [0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 8, 0, 0, 0, 7, 0, 9, 0],
    [6, 0, 2, 0, 0, 0, 5, 0, 0],
    [0, 7, 0, 0, 6, 0, 0, 0, 0],
    [0, 0, 0, 9, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 4, 0],
    [0, 0, 5, 0, 0, 0, 6, 0, 3],
    [0, 9, 0, 4, 0, 0, 0, 7, 0],
    [0, 0, 6, 0, 0, 0, 0, 0, 0]],
    dtype=np.int8
)

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

board_to_solve = hard_board
pretty_print(board_to_solve, original_board=board_to_solve)
tic = time()
solved = functions.run_backtrack(board_to_solve)
tac = time()
print(f'Done in {tac-tic:1.5f}s')
pretty_print(solved, original_board=board_to_solve)