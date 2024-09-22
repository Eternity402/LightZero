from libc.stdint cimport int32_t
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def legal_actions_cython(int32_t[:, :] board, int32_t player):
    cdef list legal_actions = []
    cdef int i, j, x, y, dx, dy
    cdef int32_t opponent = 3 - player  # opponent's num
    cdef int32_t n = 8  # othello board size 8x8
    
    # direction vector for searching
    # cdef int32_t directions[8][2] = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    cdef list directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # search all board 
    for i in range(n):
        for j in range(n):
            if board[i, j] == 0:  # search if empty
                for dx, dy in directions:
                    x, y = i + dx, j + dy
                    has_opponent_between = False
                    
                    # search if loc is in the board & opponent side
                    while 0 <= x < n and 0 <= y < n and board[x, y] == opponent:
                        x += dx
                        y += dy
                        has_opponent_between = True
                    
                    # legal action 
                    if has_opponent_between and 0 <= x < n and 0 <= y < n and board[x, y] == player:
                        legal_actions.append(i * n + j)
                        break
    
    return legal_actions