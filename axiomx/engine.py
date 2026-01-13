"""
Enhanced Chess Engine with Tensor-Based Evaluation
===================================================

Integrates tensor gradient field evaluation with improved minimax search.
Key improvements:
1. Tensor-based position evaluation
2. Enhanced move ordering with killer moves
3. Transposition table (Zobrist hashing)
4. Quiescence search for tactical positions
5. Iterative deepening with time management
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from functools import lru_cache
import time

# Import tensor engine
from tensor_engine import TensorChessEngine


# =============================================================================
# ZOBRIST HASHING FOR TRANSPOSITION TABLE
# =============================================================================

# Precomputed random numbers for Zobrist hashing
np.random.seed(42)  # Reproducible
ZOBRIST_PIECES = np.random.randint(0, 2**64, size=(8, 8, 12), dtype=np.uint64)
ZOBRIST_SIDE = np.random.randint(0, 2**64, dtype=np.uint64)


def zobrist_hash(board: List[List[Optional[Dict]]], side_to_move: str) -> int:
    """
    Compute Zobrist hash for transposition table.
    
    Fast hashing using XOR operations on precomputed random numbers.
    """
    hash_value = 0
    
    piece_to_index = {
        ('pawn', 'white'): 0, ('knight', 'white'): 1,
        ('bishop', 'white'): 2, ('rook', 'white'): 3,
        ('queen', 'white'): 4, ('king', 'white'): 5,
        ('pawn', 'black'): 6, ('knight', 'black'): 7,
        ('bishop', 'black'): 8, ('rook', 'black'): 9,
        ('queen', 'black'): 10, ('king', 'black'): 11,
    }
    
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece:
                piece_index = piece_to_index[(piece['type'], piece['color'])]
                hash_value ^= ZOBRIST_PIECES[row, col, piece_index]
    
    if side_to_move == 'black':
        hash_value ^= ZOBRIST_SIDE
    
    return hash_value


# =============================================================================
# TRANSPOSITION TABLE
# =============================================================================

class TranspositionTable:
    """
    Cache of previously evaluated positions.
    
    Stores: {hash -> (depth, score, best_move, node_type)}
    node_type: 'exact', 'lower', 'upper'
    """
    
    def __init__(self, max_size: int = 100000):
        self.table: Dict[int, Tuple[int, float, Optional[Tuple], str]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def store(self, hash_key: int, depth: int, score: float, 
              best_move: Optional[Tuple], node_type: str):
        """Store position evaluation."""
        # If table full, clear oldest entries
        if len(self.table) >= self.max_size:
            # Simple replacement: clear 20% of table
            keys_to_remove = list(self.table.keys())[:self.max_size // 5]
            for key in keys_to_remove:
                del self.table[key]
        
        # Always replace or store if deeper search
        if hash_key not in self.table or self.table[hash_key][0] <= depth:
            self.table[hash_key] = (depth, score, best_move, node_type)
    
    def probe(self, hash_key: int, depth: int, alpha: float, beta: float) -> Optional[Tuple[float, Optional[Tuple]]]:
        """
        Probe transposition table.
        
        Returns: (score, best_move) if usable, None otherwise
        """
        if hash_key not in self.table:
            self.misses += 1
            return None
        
        stored_depth, stored_score, best_move, node_type = self.table[hash_key]
        
        # Only use if stored depth >= current depth
        if stored_depth < depth:
            self.misses += 1
            return None
        
        # Check if stored score is usable
        if node_type == 'exact':
            self.hits += 1
            return (stored_score, best_move)
        elif node_type == 'lower' and stored_score >= beta:
            self.hits += 1
            return (stored_score, best_move)
        elif node_type == 'upper' and stored_score <= alpha:
            self.hits += 1
            return (stored_score, best_move)
        
        self.misses += 1
        return None
    
    def clear(self):
        """Clear transposition table."""
        self.table.clear()
        self.hits = 0
        self.misses = 0


# =============================================================================
# ORIGINAL ENGINE FUNCTIONS (for compatibility)
# =============================================================================

def initialize_board():
    board = [[None for _ in range(8)] for _ in range(8)]
    for i in range(8):
        board[1][i] = {'type': 'pawn', 'color': 'black', 'moved': False}
        board[6][i] = {'type': 'pawn', 'color': 'white', 'moved': False}
    board[0][0] = board[0][7] = {'type': 'rook', 'color': 'black', 'moved': False}
    board[7][0] = board[7][7] = {'type': 'rook', 'color': 'white', 'moved': False}
    board[0][1] = board[0][6] = {'type': 'knight', 'color': 'black', 'moved': False}
    board[7][1] = board[7][6] = {'type': 'knight', 'color': 'white', 'moved': False}
    board[0][2] = board[0][5] = {'type': 'bishop', 'color': 'black', 'moved': False}
    board[7][2] = board[7][5] = {'type': 'bishop', 'color': 'white', 'moved': False}
    board[0][3] = {'type': 'queen', 'color': 'black', 'moved': False}
    board[7][3] = {'type': 'queen', 'color': 'white', 'moved': False}
    board[0][4] = {'type': 'king', 'color': 'black', 'moved': False}
    board[7][4] = {'type': 'king', 'color': 'white', 'moved': False}
    return board

def copy_board(board):
    new_board = []
    for row in board:
        new_row = []
        for piece in row:
            if piece:
                new_row.append(piece.copy())
            else:
                new_row.append(None)
        new_board.append(new_row)
    return new_board

def is_path_clear(board, from_row, from_col, to_row, to_col):
    row_dir = 0 if to_row == from_row else (1 if to_row > from_row else -1)
    col_dir = 0 if to_col == from_col else (1 if to_col > from_col else -1)
    curr_row, curr_col = from_row + row_dir, from_col + col_dir
    while curr_row != to_row or curr_col != to_col:
        if board[curr_row][curr_col] is not None:
            return False
        curr_row += row_dir
        curr_col += col_dir
    return True

def is_valid_move(board, from_row, from_col, to_row, to_col, en_passant=None, check_castling=True):
    if to_row < 0 or to_row > 7 or to_col < 0 or to_col > 7:
        return False
    piece = board[from_row][from_col]
    if not piece:
        return False
    target = board[to_row][to_col]
    if target and target['color'] == piece['color']:
        return False
    row_diff = to_row - from_row
    col_diff = to_col - from_col
    abs_row_diff = abs(row_diff)
    abs_col_diff = abs(col_diff)
    
    if piece['type'] == 'pawn':
        direction = -1 if piece['color'] == 'white' else 1
        start_row = 6 if piece['color'] == 'white' else 1
        if col_diff == 0:
            if row_diff == direction and target is None:
                return True
            if from_row == start_row and row_diff == 2 * direction and target is None:
                mid_row = from_row + direction
                if board[mid_row][from_col] is None:
                    return True
        elif abs_col_diff == 1 and row_diff == direction:
            if target and target['color'] != piece['color']:
                return True
            if en_passant and (to_row, to_col) == en_passant:
                return True
        return False
    elif piece['type'] == 'rook':
        if row_diff == 0 or col_diff == 0:
            return is_path_clear(board, from_row, from_col, to_row, to_col)
    elif piece['type'] == 'knight':
        return (abs_row_diff == 2 and abs_col_diff == 1) or (abs_row_diff == 1 and abs_col_diff == 2)
    elif piece['type'] == 'bishop':
        if abs_row_diff == abs_col_diff:
            return is_path_clear(board, from_row, from_col, to_row, to_col)
    elif piece['type'] == 'queen':
        if row_diff == 0 or col_diff == 0 or abs_row_diff == abs_col_diff:
            return is_path_clear(board, from_row, from_col, to_row, to_col)
    elif piece['type'] == 'king':
        if abs_row_diff <= 1 and abs_col_diff <= 1:
            return True
        if check_castling and not piece.get('moved', False) and row_diff == 0 and abs_col_diff == 2:
            if not is_in_check(board, piece['color']):
                if col_diff == 2:
                    rook = board[from_row][7]
                    if rook and rook['type'] == 'rook' and not rook.get('moved', False):
                        if board[from_row][5] is None and board[from_row][6] is None:
                            if not is_square_attacked(board, from_row, 5, 'black' if piece['color'] == 'white' else 'white'):
                                if not is_square_attacked(board, from_row, 6, 'black' if piece['color'] == 'white' else 'white'):
                                    return True
                elif col_diff == -2:
                    rook = board[from_row][0]
                    if rook and rook['type'] == 'rook' and not rook.get('moved', False):
                        if board[from_row][1] is None and board[from_row][2] is None and board[from_row][3] is None:
                            if not is_square_attacked(board, from_row, 3, 'black' if piece['color'] == 'white' else 'white'):
                                if not is_square_attacked(board, from_row, 2, 'black' if piece['color'] == 'white' else 'white'):
                                    return True
    return False

def find_king(board, color):
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece and piece['type'] == 'king' and piece['color'] == color:
                return (row, col)
    return None

def is_square_attacked(board, row, col, by_color):
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece and piece['color'] == by_color:
                if is_valid_move(board, r, c, row, col, None, check_castling=False):
                    return True
    return False

def is_in_check(board, color):
    king_pos = find_king(board, color)
    if not king_pos:
        return False
    opponent_color = 'black' if color == 'white' else 'white'
    return is_square_attacked(board, king_pos[0], king_pos[1], opponent_color)

def would_be_in_check(board, from_row, from_col, to_row, to_col, color, en_passant=None):
    temp_board = copy_board(board)
    if en_passant and (to_row, to_col) == en_passant:
        piece = temp_board[from_row][from_col]
        if piece and piece['type'] == 'pawn':
            captured_pawn_row = from_row
            temp_board[captured_pawn_row][to_col] = None
    temp_board[to_row][to_col] = temp_board[from_row][from_col]
    temp_board[from_row][from_col] = None
    return is_in_check(temp_board, color)

def get_all_valid_moves(board, color, en_passant=None):
    moves = []
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece and piece['color'] == color:
                for to_row in range(8):
                    for to_col in range(8):
                        if is_valid_move(board, row, col, to_row, to_col, en_passant):
                            if not would_be_in_check(board, row, col, to_row, to_col, color, en_passant):
                                moves.append((row, col, to_row, to_col))
    return moves


# =============================================================================
# TENSOR-BASED EVALUATION (replaces old evaluate_board)
# =============================================================================

# Global tensor engine instance
_tensor_engine = None

def get_tensor_engine():
    """Lazy initialization of tensor engine."""
    global _tensor_engine
    if _tensor_engine is None:
        _tensor_engine = TensorChessEngine()
    return _tensor_engine


def evaluate_board(board, move_count=0):
    """
    Tensor-based evaluation function.
    
    Returns:
        score > 0: Black advantage
        score < 0: White advantage
    """
    engine = get_tensor_engine()
    return engine.evaluate_position(board, move_count)


# =============================================================================
# ENHANCED MOVE ORDERING
# =============================================================================

class KillerMoves:
    """Store killer moves for move ordering."""
    
    def __init__(self):
        self.killers = {}  # {depth: [move1, move2]}
    
    def add(self, depth: int, move: Tuple):
        """Add a killer move at this depth."""
        if depth not in self.killers:
            self.killers[depth] = []
        
        if move not in self.killers[depth]:
            self.killers[depth].insert(0, move)
            if len(self.killers[depth]) > 2:
                self.killers[depth].pop()
    
    def get(self, depth: int) -> List[Tuple]:
        """Get killer moves at this depth."""
        return self.killers.get(depth, [])


# Global killer moves
_killer_moves = KillerMoves()


def score_move(board, move, depth=0):
    """
    Enhanced move scoring for move ordering.
    
    Priority:
    1. Captures (MVV-LVA)
    2. Killer moves
    3. Center control
    4. Piece development
    """
    from_row, from_col, to_row, to_col = move
    score = 0
    
    piece = board[from_row][from_col]
    target = board[to_row][to_col]
    
    # 1. Captures (Most Valuable Victim - Least Valuable Attacker)
    if target:
        piece_values = {'pawn': 1, 'knight': 3, 'bishop': 3, 'rook': 5, 'queen': 9, 'king': 0}
        victim_value = piece_values.get(target['type'], 0)
        attacker_value = piece_values.get(piece['type'], 0)
        score += 1000 + 10 * victim_value - attacker_value
    
    # 2. Killer moves
    if move in _killer_moves.get(depth):
        score += 900
    
    # 3. Center control
    center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
    if (to_row, to_col) in center_squares:
        score += 50
    
    # 4. Piece development (from back rank)
    if piece['type'] in ['knight', 'bishop']:
        if piece['color'] == 'white' and from_row == 7:
            score += 30
        elif piece['color'] == 'black' and from_row == 0:
            score += 30
    
    # 5. Pawn advancement
    if piece['type'] == 'pawn':
        if piece['color'] == 'white':
            score += (7 - to_row) * 2  # Closer to promotion
        else:
            score += to_row * 2
    
    return score


def order_moves(board, moves, depth=0):
    """
    Order moves for alpha-beta efficiency.
    
    Better move ordering = more cutoffs = faster search.
    """
    scored_moves = [(score_move(board, move, depth), move) for move in moves]
    scored_moves.sort(reverse=True, key=lambda x: x[0])
    return [move for _, move in scored_moves]


# =============================================================================
# QUIESCENCE SEARCH
# =============================================================================

def quiescence(board, alpha, beta, color, en_passant=None, max_depth=4):
    """
    Quiescence search to avoid horizon effect.
    
    Searches only captures until position is "quiet".
    Prevents missing tactical sequences just beyond search depth.
    """
    # Stand pat score
    stand_pat = evaluate_board(board)
    
    if color == 'white':
        if stand_pat <= alpha:
            return alpha
        if stand_pat < beta:
            beta = stand_pat
    else:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
    
    if max_depth <= 0:
        return stand_pat
    
    # Generate only capture moves
    moves = get_all_valid_moves(board, color, en_passant)
    capture_moves = [m for m in moves if board[m[2]][m[3]] is not None]
    
    if not capture_moves:
        return stand_pat
    
    # Order captures by MVV-LVA
    capture_moves = order_moves(board, capture_moves)
    
    for move in capture_moves:
        temp_board = copy_board(board)
        
        # Handle en passant
        if en_passant and (move[2], move[3]) == en_passant:
            temp_board[move[0]][move[3]] = None
        
        temp_board[move[2]][move[3]] = temp_board[move[0]][move[1]]
        temp_board[move[0]][move[1]] = None
        
        opponent = 'white' if color == 'black' else 'black'
        score = quiescence(temp_board, alpha, beta, opponent, None, max_depth - 1)
        
        if color == 'black':
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        else:
            if score <= alpha:
                return alpha
            if score < beta:
                beta = score
    
    return alpha if color == 'black' else beta


# =============================================================================
# ENHANCED MINIMAX WITH TRANSPOSITION TABLE
# =============================================================================

# Global transposition table
_tt = TranspositionTable(max_size=200000)


def minimax(board, depth, alpha, beta, maximizing, en_passant=None, 
           move_count=0, root_call=True):
    """
    Enhanced minimax with:
    1. Transposition table
    2. Killer moves
    3. Quiescence search at leaves
    4. Null move pruning (future enhancement)
    """
    # Compute hash for transposition table
    side = 'black' if maximizing else 'white'
    board_hash = zobrist_hash(board, side)
    
    # Probe transposition table
    tt_result = _tt.probe(board_hash, depth, alpha, beta)
    if tt_result is not None and not root_call:
        return tt_result[0], tt_result[1]
    
    # Leaf node: quiescence search
    if depth == 0:
        q_score = quiescence(board, alpha, beta, side, en_passant)
        _tt.store(board_hash, 0, q_score, None, 'exact')
        return q_score, None
    
    # Generate and order moves
    color = 'black' if maximizing else 'white'
    moves = get_all_valid_moves(board, color, en_passant)
    
    if not moves:
        # Checkmate or stalemate
        if is_in_check(board, color):
            score = -100000 if maximizing else 100000
            _tt.store(board_hash, depth, score, None, 'exact')
            return score, None
        _tt.store(board_hash, depth, 0, None, 'exact')
        return 0, None
    
    moves = order_moves(board, moves, depth)
    best_move = moves[0]
    node_type = 'upper' if maximizing else 'lower'
    
    if maximizing:
        max_eval = float('-inf')
        for move in moves:
            temp_board = copy_board(board)
            
            # Handle en passant
            new_en_passant = None
            if en_passant and (move[2], move[3]) == en_passant:
                temp_board[move[0]][move[3]] = None
            
            # Handle pawn double move
            piece = temp_board[move[0]][move[1]]
            if piece['type'] == 'pawn' and abs(move[2] - move[0]) == 2:
                new_en_passant = ((move[0] + move[2]) // 2, move[3])
            
            temp_board[move[2]][move[3]] = temp_board[move[0]][move[1]]
            temp_board[move[0]][move[1]] = None
            
            eval_score, _ = minimax(temp_board, depth - 1, alpha, beta, False, 
                                   new_en_passant, move_count + 1, False)
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                node_type = 'exact'
            
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                # Beta cutoff - killer move
                _killer_moves.add(depth, move)
                node_type = 'lower'
                break
        
        _tt.store(board_hash, depth, max_eval, best_move, node_type)
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in moves:
            temp_board = copy_board(board)
            
            new_en_passant = None
            if en_passant and (move[2], move[3]) == en_passant:
                temp_board[move[0]][move[3]] = None
            
            piece = temp_board[move[0]][move[1]]
            if piece['type'] == 'pawn' and abs(move[2] - move[0]) == 2:
                new_en_passant = ((move[0] + move[2]) // 2, move[3])
            
            temp_board[move[2]][move[3]] = temp_board[move[0]][move[1]]
            temp_board[move[0]][move[1]] = None
            
            eval_score, _ = minimax(temp_board, depth - 1, alpha, beta, True, 
                                   new_en_passant, move_count + 1, False)
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
                node_type = 'exact'
            
            beta = min(beta, eval_score)
            if beta <= alpha:
                _killer_moves.add(depth, move)
                node_type = 'upper'
                break
        
        _tt.store(board_hash, depth, min_eval, best_move, node_type)
        return min_eval, best_move


def get_ai_move(board, en_passant=None, color='black', move_count=0):
    """
    Main AI move generation with iterative deepening.
    
    Searches progressively deeper until time limit or target depth reached.
    """
    maximizing = (color == 'black')
    
    # Start with depth 3, can go deeper if time allows
    target_depth = 3
    best_move = None
    
    for current_depth in range(1, target_depth + 1):
        _, move = minimax(board, current_depth, float('-inf'), float('inf'), 
                         maximizing, en_passant, move_count, root_call=True)
        if move:
            best_move = move
    
    return best_move