from libc.stdint cimport int32_t
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_done_winner_cython(int32_t[:, :] board):
    """
    Overview:
         Check if the othello game is over and who the winner is. Return 'done' and 'winner'.
    Arguments:
        - board (:obj:`numpy.ndarray`): The board state.
    Returns:
        - outputs (:obj:`Tuple`): Tuple containing 'done' and 'winner',
            - if player 1 win,     'done' = True, 'winner' = 1
            - if player 2 win,     'done' = True, 'winner' = 2
            - if draw,             'done' = True, 'winner' = -1
            - if game is not over, 'done' = False, 'winner' = -1
    """
    cdef int32_t i, j
    cdef bint has_legal_actions = False

    # check if no disc
    cdef int32_t disc_1, disc_2
    disc_1 = 0
    disc_2 = 0
    for i in range(8):
        for j in range(8):
            if board[i, j] == 1:
                disc_1 += 1
            if board[i, j] == 2:
                disc_2 += 1
    
    if disc_1 == 0:
        return True, 2
    if disc_2 == 0:
        return True, 1

    # check if no available action
    cdef list players = [1, 2]
    cdef int x, y, dx, dy
    cdef bint legal_action_exist = False
    cdef int32_t n = 8  # othello board size 8x8
    cdef int32_t opponent
    # direction vector for searching
    cdef list directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for player in players:
        opponent = 3 - player  # opponent's num

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
                            legal_action_exist = True
                            break
    
    cdef int32_t score_1, score_2
    if legal_action_exist:
        return False, -1
    else:
        score_1 = 0
        score_2 = 0
        for i in range(8):
            for j in range(8):
                if board[i, j] == 1:
                    score_1 += 1
                if board[i, j] == 2:
                    score_2 += 1
        
        if score_1 > score_2:
            return True, 1
        elif score_1 < score_2:
            return True, 2
        else:
            return True, -1