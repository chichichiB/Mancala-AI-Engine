###############################################################################
# This file implements various alpha-beta pruning agents.
#
# CSC 384 Assignment 2 Starter Code
# version 2.0
###############################################################################
from wrapt_timeout_decorator import timeout

from mancala_game import play_move
from utils import *


def alphabeta_max_basic(board, curr_player, alpha, beta, heuristic_func):
    """
    Perform Alpha-Beta Search for MAX player.
    Return the best move and the estimated minimax value.

    If the board is a terminal state,
    return None as the best move and the heuristic value of the board as the best value.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function
    :return the best move and its minimax value.
    """
    if sum(board.pockets[TOP]) == 0 or sum(board.pockets[BOTTOM]) == 0:
        return None, heuristic_func(board, curr_player)

    best_move = None
    best_value = float('-inf')

    for move in board.get_possible_moves(curr_player):
        next_state = play_move(board, curr_player, move)
        _, value = alphabeta_min_basic(next_state, curr_player, alpha, beta, heuristic_func)
        if value > alpha:
            alpha = value
            best_value = alpha
            best_move = move
        else:
            continue
        if alpha >= beta:
            break
    return best_move, best_value

def alphabeta_min_basic(board, curr_player, alpha, beta, heuristic_func):
    """
    Perform Alpha-Beta Search for MIN player.
    Return the best move and the estimated minimax value.

    If the board is a terminal state,
    return None as the best move and the heuristic value of the board as the best value.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function
    :return the best move and its minimax value.
    """

    if sum(board.pockets[TOP]) == 0 or sum(board.pockets[BOTTOM]) == 0:
        return None, heuristic_func(board, curr_player)

    best_move = None
    best_value = float('inf')

    for move in board.get_possible_moves(get_opponent(curr_player)):
        next_state = play_move(board, get_opponent(curr_player), move)
        _, value = alphabeta_max_basic(next_state, curr_player, alpha, beta, heuristic_func)

        if value < beta:
            beta = value
            best_value = beta
            best_move = move
        else:
            continue
        if beta <= alpha:
            break
    return best_move, best_value

def alphabeta_max_limit(board, curr_player, alpha, beta, heuristic_func, depth_limit):
    """
    Perform Alpha-Beta Search for MAX player up to the given depth limit.
    Return the best move and the estimated minimax value.

    If the board is a terminal state,
    return None as the best move and the heuristic value of the board as the best value.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :return the best move and its estimated minimax value.
    """

    if sum(board.pockets[TOP]) == 0 or sum(board.pockets[BOTTOM]) == 0 or depth_limit == 0:
        return None, heuristic_func(board, curr_player)

    depth_limit -= 1
    best_move = None
    best_value = float('-inf')

    for move in board.get_possible_moves(curr_player):
        next_state = play_move(board, curr_player, move)
        _, value = alphabeta_min_limit(next_state, curr_player, alpha, beta, heuristic_func, depth_limit)
        if value > alpha:
            alpha = value
            best_value = alpha
            best_move = move
        else:
            continue
        if alpha >= beta:
            break
    return best_move, best_value

def alphabeta_min_limit(board, curr_player, alpha, beta, heuristic_func, depth_limit):
    """
    Perform Alpha-Beta Search for MIN player up to the given depth limit.
    Return the best move and the estimated minimax value.

    If the board is a terminal state,
    return None as the best move and the heuristic value of the board as the best value.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :return the best move and its estimated minimax value.
    """

    if sum(board.pockets[TOP]) == 0 or sum(board.pockets[BOTTOM]) == 0 or depth_limit == 0:
        return None, heuristic_func(board, curr_player)

    depth_limit -= 1
    best_move = None
    best_value = float('inf')

    for move in board.get_possible_moves(get_opponent(curr_player)):
        next_state = play_move(board, get_opponent(curr_player), move)
        _, value = alphabeta_max_limit(next_state, curr_player, alpha, beta, heuristic_func, depth_limit)
        if value < beta:
            beta = value
            best_value = beta
            best_move = move
        else:
            continue
        if beta <= alpha:
            break
    return best_move, best_value

EXACT, LOWER, UPPER = 0, 1, 2

def _tt_key(board, curr_player, depth_limit):
    return (board, curr_player, depth_limit)

def _tt_probe(tt, key, alpha, beta):
    entry = tt.get(key)
    if not entry:
        return alpha, beta, None, None
    depth, flag, value, best_move = entry
    if flag == EXACT:
        return alpha, beta, best_move, value
    if flag == LOWER:
        alpha = max(alpha, value)
    elif flag == UPPER:
        beta = min(beta, value)
    return alpha, beta, best_move, None

def _tt_store(tt, key, depth, value, best_move, alpha0, beta0):
    if value <= alpha0:
        flag = UPPER
    elif value >= beta0:
        flag = LOWER
    else:
        flag = EXACT
    tt[key] = (depth, flag, value, best_move)

def order_moves(board, curr_player, moves, heuristic_func, maximize = True):

    scored = []
    for m in moves:
        next_state = play_move(board, curr_player, m)
        s = heuristic_func(next_state, curr_player)
        scored.append((m, next_state, s))

    scored.sort(key = lambda t: t[2], reverse=maximize)
    return [(m, next_state) for (m, next_state, _) in scored]

def alphabeta_max_limit_opt(board, curr_player, alpha, beta, heuristic_func, depth_limit, optimizations):
    """
    Perform Alpha-Beta Search for MAX player
    up to the given depth limit and with additional optimizations.
    Return the best move and the estimated minimax value.

    If the board is a terminal state,
    return None as the best move and the heuristic value of the board as the best value.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :param optimizations: a dictionary to contain any data structures for optimizations.
        You can use a dictionary called "cache" to implement caching.
    :return the best move and its estimated minimax value.
    """

    if sum(board.pockets[TOP]) == 0 or sum(board.pockets[BOTTOM]) == 0 or depth_limit == 0:
        return None, heuristic_func(board, curr_player)
    tt = optimizations.get("cache", {})

    key = _tt_key(board, curr_player, depth_limit)
    a0, b0 = alpha, beta
    alpha, beta, pv_move, exact = _tt_probe(tt, key, alpha, beta)
    if exact is not None:
        return pv_move, exact
    if alpha >= beta:

        _tt_store(tt, key, depth_limit, alpha, pv_move, a0, b0)
        return pv_move, alpha

    moves = board.get_possible_moves(curr_player)

    moves = order_moves(board, curr_player, moves, heuristic_func, maximize = True)

    depth_limit -= 1
    best_move = None
    best_value = float('-inf')

    for move, next_state in moves:
        _, value = alphabeta_min_limit_opt(next_state, curr_player, alpha, beta, heuristic_func, depth_limit, optimizations)
        if value > alpha:
            alpha = value
            best_value = alpha
            best_move = move
        else:
            continue
        if alpha >= beta:
            break
    _tt_store(tt, key, depth_limit, best_value, best_move, a0, b0)
    return best_move, best_value

def alphabeta_min_limit_opt(board, curr_player, alpha, beta, heuristic_func, depth_limit, optimizations):
    """
    Perform Alpha-Beta Pruning for MIN player
    up to the given depth limit and with additional optimizations.
    Return the best move and the estimated minimax value.

    If the board is a terminal state,
    return None as the best move and the heuristic value of the board as the best value.

    :param board: the current board
    :param curr_player: the current player
    :param alpha: current alpha value
    :param beta: current beta value
    :param heuristic_func: the heuristic function
    :param depth_limit: the depth limit
    :param optimizations: a dictionary to contain any data structures for optimizations.
        You can use a dictionary called "cache" to implement caching.
    :return the best move and its estimated minimax value.
    """

    if sum(board.pockets[TOP]) == 0 or sum(board.pockets[BOTTOM]) == 0 or depth_limit == 0:
        return None, heuristic_func(board, curr_player)
    tt = optimizations.get("cache", {})

    key = _tt_key(board, get_opponent(curr_player), depth_limit)
    a0, b0 = alpha, beta
    alpha, beta, pv_move, exact = _tt_probe(tt, key, alpha, beta)
    if exact is not None:
        return pv_move, exact
    if alpha >= beta:
        _tt_store(tt, key, depth_limit, beta, pv_move, a0, b0)
        return pv_move, beta

    moves = board.get_possible_moves(get_opponent(curr_player))

    moves = order_moves(board, get_opponent(curr_player), moves, heuristic_func, maximize = False)

    depth_limit -= 1
    best_move = None
    best_value = float('inf')

    for move, next_state in moves:
        _, value = alphabeta_max_limit_opt(next_state, curr_player, alpha, beta, heuristic_func, depth_limit, optimizations)
        if value < beta:
            beta = value
            best_value = beta
            best_move = move
        else:
            continue
        if beta <= alpha:
            break

    _tt_store(tt, key, depth_limit, best_value, best_move, a0, b0)
    return best_move, best_value


###############################################################################
## DO NOT MODIFY THE CODE BELOW.
###############################################################################

@timeout(TIMEOUT, timeout_exception=AiTimeoutError)
def run_alphabeta(curr_board, player, limit, optimizations, hfunc):
    if optimizations is not None:
        opt = True
    else:
        opt = False

    alpha = float("-Inf")
    beta = float("Inf")
    if opt:
        move, value = alphabeta_max_limit_opt(curr_board, player, alpha, beta, hfunc, limit, optimizations)
    elif limit >= 0:
        move, value = alphabeta_max_limit(curr_board, player, alpha, beta, hfunc, limit)
    else:
        move, value = alphabeta_max_basic(curr_board, player, alpha, beta, hfunc)

    return move, value

