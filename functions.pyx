# cython: language_level=2
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cimport numpy as cnp
import numpy as np

cnp.import_array()

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

cdef cnp.int8_t[:,:] BOARD = np.array([
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

cdef Py_ssize_t BOARD_CONSTANT = 3

cdef find_collections_to_check(cnp.int8_t i,
                               cnp.int8_t j):
    '''
    This function find all the collections that an entry is part of
    :param i: row of entry
    :param j: col of entry
    :return: the iterators of all collections that entry (i, j) is a part of
    '''
    # rowwise
    cdef (cnp.int8_t, cnp.int8_t) rows_rowwise = (i, i+1)
    cdef (cnp.int8_t, cnp.int8_t) cols_rowwise = (0, BOARD_CONSTANT**2)
    cdef ((cnp.int8_t, cnp.int8_t), (cnp.int8_t, cnp.int8_t)) rowwise = (rows_rowwise, cols_rowwise)

    # colwise
    cdef (cnp.int8_t, cnp.int8_t) rows_colwise = (0, BOARD_CONSTANT**2)
    cdef (cnp.int8_t, cnp.int8_t) cols_colwise = (j, j+1)
    cdef ((cnp.int8_t, cnp.int8_t), (cnp.int8_t, cnp.int8_t)) colwise = (rows_colwise, cols_colwise)

    # squarewise
    # rows that are in the square
    cdef cnp.int8_t row_start = BOARD_CONSTANT * (i // BOARD_CONSTANT)
    cdef cnp.int8_t row_end = row_start + BOARD_CONSTANT
    cdef (cnp.int8_t, cnp.int8_t) rows_squarewise = (row_start, row_end)

    # cols that are in the square
    cdef cnp.int8_t col_start = BOARD_CONSTANT * (j // BOARD_CONSTANT)
    cdef cnp.int8_t col_end = col_start + BOARD_CONSTANT
    cdef (cnp.int8_t, cnp.int8_t) cols_squarewise = (col_start, col_end)
    cdef ((cnp.int8_t, cnp.int8_t), (cnp.int8_t, cnp.int8_t)) squarewise = (rows_squarewise, cols_squarewise)

    return rowwise, colwise, squarewise

cdef dict all_collections():
    cdef Py_ssize_t i, j
    cdef dict collections_to_check = {}
    for i in range(BOARD_CONSTANT**2):
        for j in range(BOARD_CONSTANT**2):
            collections_to_check[(i,j)] = find_collections_to_check(i,j)
    return collections_to_check

cdef dict collections_to_check = all_collections()

cdef bint check_valid_collection(cnp.int8_t[:,:] board,
                                 ((cnp.int8_t, cnp.int8_t), (cnp.int8_t, cnp.int8_t)) collection):
    '''
    This function checks if a collection is valid
    :param board: the board to check
    :param collection: the iterators to check
    :return: boolean value that indicates if the collection that was checked is valid (no duplicate numbers)
    '''
    cdef list appeared_numbers = [0]*BOARD_CONSTANT**2
    cdef Py_ssize_t i, j
    cdef cnp.int8_t entry

    cdef (cnp.int8_t, cnp.int8_t) row_iterator = collection[0]
    cdef (cnp.int8_t, cnp.int8_t) col_iterator = collection[1]

    for i in range(row_iterator[0], row_iterator[1]):
        for j in range(col_iterator[0], col_iterator[1]):
            entry = board[i,j]
            if entry == 0:
                continue
            if appeared_numbers[entry - 1] > 0:
                return False
            appeared_numbers[entry - 1] += 1
    return True

cdef bint check_valid(cnp.int8_t[:,:] board,
                      (cnp.int8_t, cnp.int8_t) entry):
    '''
    This function checks all the collections that entry is a part of
    :param board: the board to check
    :param entry: the entry of interest
    :return: boolean value that indicates whether the new board is valid
    '''
    cdef cnp.int8_t i, j
    i = entry[0]
    j = entry[1]
    #print(f'Entry: {entry}')

    cdef Py_ssize_t k
    collections = collections_to_check[(i,j)]
    cdef ((cnp.int8_t, cnp.int8_t), (cnp.int8_t, cnp.int8_t)) collection

    cdef Py_ssize_t length = len(collections)

    for k in range(length):
        collection = collections[k]
        if not check_valid_collection(board, collection):
            return False
    return True

# This is the array that is returned if there is no valid solution
cdef cnp.int8_t[:,:] empty_array = np.zeros_like(EMPTY_BOARD)
empty_array[0,0] = -1

cdef (Py_ssize_t, Py_ssize_t) find_last_entry_to_fill(cnp.int8_t[:,:] board):
    '''
    This function is used to check if the looping arrived at the last entry
    :param board: the board
    :return: the last entry
    '''
    cdef (Py_ssize_t, Py_ssize_t) last_entry
    cdef cnp.int8_t entry
    cdef Py_ssize_t i, j
    for i in range(len(board)):
        row = board[i]
        for j in range(len(row)):
            entry = row[j]
            if entry == 0:
                last_entry = (i,j)
    return last_entry

cdef (Py_ssize_t, Py_ssize_t) last_entry = find_last_entry_to_fill(BOARD)

cdef cnp.int8_t[:,:] backtrack(cnp.int8_t[:,:] board):
    '''
    Do the actual backtracking using the latest board
    :param board: the board
    :return: the updated board or an empty board where [0,0] = -1. This means: no valid possibility
    '''
    new_board = board.copy()
    cdef Py_ssize_t i, j
    cdef (Py_ssize_t, Py_ssize_t) current_entry
    cdef cnp.int8_t[:,:] updated_board
    cdef cnp.int8_t entry, possible_value
    for i in range(len(board)):
        row = board[i]
        for j in range(len(row)):
            entry = row[j]
            if entry == 0:
                for possible_value in range(1,10):  # Try all values
                    new_board[i,j] = possible_value
                    if check_valid(new_board, (i,j)):  # Check if they're valid
                        updated_board = backtrack(new_board)
                        if updated_board[0,0] != -1:
                            return updated_board
                        else:
                            continue
                current_entry = (i,j)
                if not (current_entry[0] == last_entry[0] and current_entry[1] == last_entry[1]):
                    return empty_array
    return new_board

cpdef run_backtrack(cnp.int8_t[:,:] board):
    return backtrack(board)