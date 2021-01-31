# cython: language_level=2
# cython: profile=True
# cython: boundscheck=False
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cimport numpy as cnp
import numpy as np

cnp.import_array()
## Initialisation
cdef cnp.int8_t[:,:] EMPTY_BOARD = np.array([
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
cdef cnp.int8_t[:,:] easy_board = np.array([
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
cdef cnp.int8_t[:,:] hard_board = np.array([
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

cdef cnp.int8_t[:,:] diagonal_board = np.array([
    [0, 3, 0, 8, 4, 0, 0, 0, 0],
    [0, 0, 0, 9, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 0, 0, 0, 0, 0],
    [2, 5, 0, 0, 0, 7, 4, 8, 0],
    [0, 0, 1, 0, 0, 0, 0, 3, 0],
    [0, 7, 3, 0, 0, 0, 0, 0, 1],
    [0, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 8, 6, 0, 0, 9, 0, 0],
    [9, 0, 0, 0, 0, 0, 0, 0, 0]],
    dtype=np.int8
)

cdef cnp.int8_t[:,:] jigsaw_board = np.array([
    [0, 0, 0, 0, 0, 0, 2, 0, 0],
    [2, 6, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 6, 0, 0, 4],
    [8, 0, 6, 4, 0, 7, 0, 3, 1],
    [0, 0, 0, 0, 0, 4, 9, 0, 0],
    [6, 0, 0, 0, 0, 2, 0, 5, 0],
    [5, 0, 2, 7, 0, 0, 1, 8, 6],
    [9, 3, 0, 0, 8, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 6, 0, 0]],
    dtype=np.int8
)

cdef Py_ssize_t[:,:,:] JIGSAW_INDICES_COLLECTION
JIGSAW_INDICES_COLLECTION = np.array([
    [[0,0], [0,1], [0,2], [0,3], [1,0], [1,1], [1,2], [2,1], [2,2]],
    [[0,4], [0,5], [0,6], [0,7], [1,3], [1,4], [1,5], [1,6], [1,7]],
    [[2,0], [2,3], [3,0], [3,1], [3,2], [3,3], [4,0], [4,1], [4,3]],
    [[2,4], [2,5], [2,6], [3,4], [3,5], [3,6], [3,7], [4,4], [4,5]],
    [[0,8], [1,8], [2,7], [2,8], [3,8], [4,6], [4,7], [4,8], [5,6]],
    [[4,2], [5,0], [5,1], [5,2], [5,3], [6,0], [6,1], [6,2], [7,0]],
    [[5,4], [5,5], [6,3], [6,4], [6,5], [7,2], [7,3], [7,4], [7,5]],
    [[8,0], [7,1], [8,1], [8,2], [8,3], [8,4], [8,5], [7,6], [8,6]],
    [[5,7], [5,8], [6,6], [6,7], [6,8], [7,7], [7,8], [8,7], [8,8]]
])

cdef Py_ssize_t BOARD_CONSTANT = 3

## Preperation
cdef Py_ssize_t[:,:] collection(Py_ssize_t row_start,
                                Py_ssize_t row_end,
                                Py_ssize_t col_start,
                                Py_ssize_t col_end):
    result = np.zeros(shape=(BOARD_CONSTANT ** 2, 2), dtype=int)
    cdef Py_ssize_t row, col, counter
    counter = 0
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            result[counter] = (row, col)
            counter += 1
    return result


cdef find_collections_to_check(Py_ssize_t row,
                               Py_ssize_t col):
    '''
    This function find all the collections that an entry is part of
    :param row: row of entry
    :param col: col of entry
    :return: a list of all collections that (i,j) i part of
    '''
    cdef list collections = []
    # rowwise
    cdef Py_ssize_t row_start, row_end, col_start, col_end
    row_start = row
    row_end = row + 1
    col_start = 0
    col_end = BOARD_CONSTANT**2
    cdef Py_ssize_t[:,:] rowwise
    rowwise = collection(row_start, row_end, col_start, col_end)
    collections.append(rowwise)

    # colwise
    row_start = 0
    row_end = BOARD_CONSTANT**2
    col_start = col
    col_end = col + 1
    cdef Py_ssize_t[:,:] colwise
    colwise = collection(row_start, row_end, col_start, col_end)
    collections.append(colwise)

    # squarewise
    cdef Py_ssize_t[:,:] squarewise
    if SQUARES:
        row_start = BOARD_CONSTANT * (row // BOARD_CONSTANT)
        row_end = row_start + BOARD_CONSTANT
        col_start = BOARD_CONSTANT * (col // BOARD_CONSTANT)
        col_end = col_start + BOARD_CONSTANT
        squarewise = collection(row_start, row_end, col_start, col_end)
        collections.append(squarewise)

    # diagonalwise
    cdef Py_ssize_t k
    cdef Py_ssize_t[:,:] diagonalwise0, diagonalwise1
    if DIAGONAL:
        if (row == col):  # If the entry is on the diagonal upper-left<-->lower-right
            result = np.zeros(shape=(BOARD_CONSTANT ** 2, 2), dtype=int)
            for k in range(BOARD_CONSTANT**2):
                result[k] = (k, k)
            diagonalwise0 = result.copy()
            collections.append(diagonalwise0)
        if (row == BOARD_CONSTANT**2-1-col):
            result = np.zeros(shape=(BOARD_CONSTANT ** 2, 2), dtype=int)
            for k in range(BOARD_CONSTANT**2):
                result[k] = (k, BOARD_CONSTANT**2-1-k)
            diagonalwise1 = result.copy()
            collections.append(diagonalwise1)

    # jigsawwise
    cdef Py_ssize_t[:,:] jigsaw_indices
    if JIGSAW:
        for k in range(BOARD_CONSTANT**2):
            jigsaw_indices = JIGSAW_INDICES_COLLECTION[k]
            if [row, col] in np.asarray(jigsaw_indices):
                jigsawwise = jigsaw_indices
                collections.append(jigsawwise)

    return collections

cdef dict all_collections():
    cdef Py_ssize_t row, col
    cdef dict collections_to_check = {}
    for row in range(BOARD_CONSTANT**2):
        for col in range(BOARD_CONSTANT**2):
            collections_to_check[(row,col)] = find_collections_to_check(row,col)
    return collections_to_check

board_to_play = jigsaw_board.copy()
cdef bint DIAGONAL = False
cdef bint JIGSAW = True
cdef bint SQUARES = False
cpdef cnp.int8_t[:,:] get_board_to_play():
    return board_to_play.copy()

cdef dict collections_to_check = all_collections()

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

cdef (Py_ssize_t, Py_ssize_t) last_entry = find_last_entry_to_fill(board_to_play)


## Running

cdef bint check_valid_collection(cnp.int8_t[:,:] board,
                                 Py_ssize_t[:,:] collection):
    '''
    This function checks if a collection is valid
    :param board: the board to check
    :param collection: the iterators to check
    :return: boolean value that indicates if the collection that was checked is valid (no duplicate numbers)
    '''
    cdef list appeared_numbers = [0]*BOARD_CONSTANT**2
    cdef Py_ssize_t i, row, col
    cdef cnp.int8_t entry

    for i in range(BOARD_CONSTANT**2):
        row = collection[i][0]
        col = collection[i][1]
        entry = board[row, col]
        if entry == 0:
            continue
        if appeared_numbers[entry - 1] > 0:
            return False
        appeared_numbers[entry - 1] += 1
    return True

cdef bint check_valid(cnp.int8_t[:,:] board,
                      (Py_ssize_t, Py_ssize_t) entry):
    '''
    This function checks all the collections that entry is a part of
    :param board: the board to check
    :param entry: the entry of interest
    :return: boolean value that indicates whether the new board is valid
    '''
    cdef Py_ssize_t row, col
    row = entry[0]
    col = entry[1]

    cdef Py_ssize_t k
    collections = collections_to_check[(row,col)]
    cdef Py_ssize_t[:,:] collection

    cdef Py_ssize_t length = len(collections)

    for k in range(length):
        collection = collections[k]
        if not check_valid_collection(board, collection):
            return False
    return True

cdef cnp.int8_t[:,:] backtrack(cnp.int8_t[:,:] board):
    '''
    Do the actual backtracking using the latest board
    :param board: the board
    :return: the updated board or an empty board where [0,0] = -1. This means: no valid possibility
    '''
    new_board = board.copy()
    cdef Py_ssize_t row, col
    cdef (Py_ssize_t, Py_ssize_t) current_entry
    cdef cnp.int8_t[:,:] updated_board
    cdef cnp.int8_t[:] row_board
    cdef cnp.int8_t entry, possible_value
    for row in range(len(board)):
        row_board = board[row]
        for col in range(len(row_board)):
            entry = row_board[col]
            if entry == 0:
                for possible_value in range(1,10):  # Try all values
                    new_board[row,col] = possible_value
                    if check_valid(new_board, (row,col)):  # Check if they're valid
                        updated_board = backtrack(new_board)
                        if updated_board[0,0] != -1:  # Check if an empty array was returned
                            return updated_board
                        else:
                            continue
                if not (row == last_entry[0] and col == last_entry[1]):  # If we're not filling the last entry
                    return empty_array
                elif not check_valid(new_board, (last_entry[0],last_entry[1])): # Check if the last entry was valid
                    return empty_array
    return new_board

cpdef run_backtrack():
    board_to_play = get_board_to_play()
    return backtrack(board_to_play)