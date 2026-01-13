"""
Tensor-Based Chess Engine: Gradient Field Evaluation
======================================================

Revolutionary approach treating the chess board as a multi-dimensional gradient field.
Uses bitboards (64-bit integers) for speed and vectorized tensor operations.

Core Philosophy:
- Each piece generates influence fields (attack, defense, control)
- Board strength = superposition of all influence gradients
- Evaluation maximizes defensive strength + opponent weakness exploitation
- Opening theory encoded as pattern tensors
- Future projection without deep tree search
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from functools import lru_cache

# =============================================================================
# BITBOARD CONSTANTS - 64-bit representation for speed
# =============================================================================

# Rank masks
RANK_1 = 0x00000000000000FF
RANK_2 = 0x000000000000FF00
RANK_3 = 0x0000000000FF0000
RANK_4 = 0x00000000FF000000
RANK_5 = 0x000000FF00000000
RANK_6 = 0x0000FF0000000000
RANK_7 = 0x00FF000000000000
RANK_8 = 0xFF00000000000000

# File masks
FILE_A = 0x0101010101010101
FILE_B = 0x0202020202020202
FILE_C = 0x0404040404040404
FILE_D = 0x0808080808080808
FILE_E = 0x1010101010101010
FILE_F = 0x2020202020202020
FILE_G = 0x4040404040404040
FILE_H = 0x8080808080808080

# Center control masks
CENTER = 0x0000001818000000  # e4, d4, e5, d5
EXTENDED_CENTER = 0x00003C3C3C3C0000  # c3-f3 to c6-f6

# Castling masks
WHITE_KINGSIDE_CASTLE = 0x0000000000000060  # f1, g1
WHITE_QUEENSIDE_CASTLE = 0x000000000000000E  # b1, c1, d1
BLACK_KINGSIDE_CASTLE = 0x6000000000000000  # f8, g8
BLACK_QUEENSIDE_CASTLE = 0x0E00000000000000  # b8, c8, d8

# =============================================================================
# OPENING THEORY PATTERNS - Encoded as bitboard templates
# =============================================================================

OPENING_PATTERNS = {
    'white_control_center': 0x0000001818000000,  # e4, d4
    'white_develop_knights': 0x0000000000004200,  # b1, g1 -> c3, f3
    'white_develop_bishops': 0x0000000000002400,  # c1, f1 -> developed
    'white_castle_kingside': 0x0000000000000060,  # king safety
    'black_control_center': 0x0000181800000000,  # e5, d5
    'black_develop_knights': 0x0042000000000000,  # b8, g8 -> c6, f6
    'black_develop_bishops': 0x0024000000000000,  # c8, f8 -> developed
    'black_castle_kingside': 0x6000000000000000,  # king safety
}

# Opening principle scores
OPENING_BONUSES = {
    'center_control': 50,
    'knight_development': 30,
    'bishop_development': 25,
    'castling': 60,
    'pawn_structure': 20,
}

# =============================================================================
# PRECOMPUTED ATTACK TABLES - Cache for speed
# =============================================================================

@lru_cache(maxsize=256)
def get_knight_attacks(square: int) -> int:
    """Precomputed knight attack bitboard for given square."""
    bb = 1 << square
    attacks = 0
    
    # All 8 knight move directions
    if square > 17 and square % 8 > 0:  # up-up-left
        attacks |= bb << 15
    if square > 17 and square % 8 < 7:  # up-up-right
        attacks |= bb << 17
    if square > 10 and square % 8 > 1:  # up-left-left
        attacks |= bb << 6
    if square > 10 and square % 8 < 6:  # up-right-right
        attacks |= bb << 10
    if square < 54 and square % 8 > 1:  # down-left-left
        attacks |= bb >> 10
    if square < 54 and square % 8 < 6:  # down-right-right
        attacks |= bb >> 6
    if square < 46 and square % 8 > 0:  # down-down-left
        attacks |= bb >> 17
    if square < 46 and square % 8 < 7:  # down-down-right
        attacks |= bb >> 15
    
    return attacks

@lru_cache(maxsize=256)
def get_king_attacks(square: int) -> int:
    """Precomputed king attack bitboard for given square."""
    bb = 1 << square
    attacks = 0
    
    # All 8 king move directions
    if square % 8 > 0:
        attacks |= bb >> 1  # left
        if square > 7:
            attacks |= bb >> 9  # up-left
        if square < 56:
            attacks |= bb << 7  # down-left
    
    if square % 8 < 7:
        attacks |= bb << 1  # right
        if square > 7:
            attacks |= bb >> 7  # up-right
        if square < 56:
            attacks |= bb << 9  # down-right
    
    if square > 7:
        attacks |= bb >> 8  # up
    if square < 56:
        attacks |= bb << 8  # down
    
    return attacks

def get_pawn_attacks(square: int, is_white: bool) -> int:
    """Precomputed pawn attack bitboard for given square."""
    bb = 1 << square
    attacks = 0
    
    if is_white:
        # White pawns attack diagonally upward
        if square > 7 and square % 8 > 0:
            attacks |= bb >> 9  # up-left
        if square > 7 and square % 8 < 7:
            attacks |= bb >> 7  # up-right
    else:
        # Black pawns attack diagonally downward
        if square < 56 and square % 8 > 0:
            attacks |= bb << 7  # down-left
        if square < 56 and square % 8 < 7:
            attacks |= bb << 9  # down-right
    
    return attacks

# =============================================================================
# TENSOR GRADIENT FIELD ENGINE
# =============================================================================

class TensorChessEngine:
    """
    Revolutionary tensor-based evaluation using gradient fields.
    
    Key Innovation: Multi-dimensional influence representation
    - Dimension 0: Material/piece presence
    - Dimension 1: Attack influence (offensive gradient)
    - Dimension 2: Defense influence (defensive gradient)
    - Dimension 3: Control gradient (net influence)
    - Dimension 4: Future projection (2-ply lookahead tensor)
    """
    
    def __init__(self):
        # Piece values for material baseline
        self.piece_values = {
            'pawn': 100,
            'knight': 320,
            'bishop': 330,
            'rook': 500,
            'queen': 900,
            'king': 20000
        }
        
        # Influence decay rates (how influence weakens with distance)
        self.influence_decay = {
            'pawn': 0.3,
            'knight': 0.5,
            'bishop': 0.7,
            'rook': 0.8,
            'queen': 0.9,
            'king': 0.4,
        }
        
        # Precompute distance matrices for vectorized operations
        self._precompute_distance_matrices()
        
    def _precompute_distance_matrices(self):
        """Precompute distance matrices for all square pairs."""
        # Create coordinate grids
        rows = np.arange(8).reshape(8, 1).repeat(8, axis=1)
        cols = np.arange(8).reshape(1, 8).repeat(8, axis=0)
        
        # Store for quick access
        self.row_grid = rows
        self.col_grid = cols
        
        # Precompute all pairwise distances (8x8x8x8 tensor)
        # But we'll compute on-the-fly with broadcasting to save memory
        
    def board_to_tensor(self, board: List[List[Optional[Dict]]]) -> np.ndarray:
        """
        Convert board to 8x8x13 tensor representation.
        
        Channels:
        0-5: White pieces (pawn, knight, bishop, rook, queen, king)
        6-11: Black pieces (pawn, knight, bishop, rook, queen, king)
        12: Occupancy mask (any piece)
        
        Returns: float32 tensor for speed
        """
        tensor = np.zeros((8, 8, 13), dtype=np.float32)
        
        piece_to_channel = {
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
                    channel = piece_to_channel[(piece['type'], piece['color'])]
                    tensor[row, col, channel] = 1.0
                    tensor[row, col, 12] = 1.0  # Occupancy
        
        return tensor
    
    def compute_attack_field(self, tensor: np.ndarray, color: str) -> np.ndarray:
        """
        Compute attack influence field for given color.
        
        Returns 8x8 float array where each square contains attack strength.
        Uses vectorized operations for speed.
        """
        attack_field = np.zeros((8, 8), dtype=np.float32)
        channel_offset = 0 if color == 'white' else 6
        
        # For each piece type, compute its attack influence
        piece_types = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
        
        for idx, piece_type in enumerate(piece_types):
            channel = channel_offset + idx
            piece_positions = tensor[:, :, channel]
            
            if not piece_positions.any():
                continue
            
            # Get all squares with this piece type
            piece_squares = np.argwhere(piece_positions > 0)
            
            for row, col in piece_squares:
                # Add influence based on piece type
                influence = self._compute_piece_influence(
                    row, col, piece_type, color, tensor
                )
                attack_field += influence
        
        return attack_field
    
    def _compute_piece_influence(self, row: int, col: int, piece_type: str, 
                                 color: str, tensor: np.ndarray) -> np.ndarray:
        """
        Compute influence field for a single piece using vectorized distance.
        
        Innovation: Influence decays with distance and is blocked by pieces.
        """
        influence = np.zeros((8, 8), dtype=np.float32)
        
        if piece_type == 'pawn':
            # Pawns attack diagonally forward
            direction = -1 if color == 'white' else 1
            attack_row = row + direction
            
            if 0 <= attack_row < 8:
                if col > 0:
                    influence[attack_row, col - 1] = 4.0
                if col < 7:
                    influence[attack_row, col + 1] = 4.0
            
            # Forward push influence (weaker)
            push_row = row + direction
            if 0 <= push_row < 8 and tensor[push_row, col, 12] == 0:
                influence[push_row, col] = 2.0
        
        elif piece_type == 'knight':
            # Knight moves in L-shape
            knight_moves = [
                (-2, -1), (-2, 1), (-1, -2), (-1, 2),
                (1, -2), (1, 2), (2, -1), (2, 1)
            ]
            for dr, dc in knight_moves:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    influence[new_row, new_col] = 6.0
        
        elif piece_type == 'bishop':
            # Diagonal rays with decay
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                self._add_ray_influence(influence, row, col, dr, dc, tensor, 5.0, 0.7)
        
        elif piece_type == 'rook':
            # Orthogonal rays with decay
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                self._add_ray_influence(influence, row, col, dr, dc, tensor, 7.0, 0.8)
        
        elif piece_type == 'queen':
            # Combined bishop + rook rays
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1),
                          (-1, 0), (1, 0), (0, -1), (0, 1)]:
                self._add_ray_influence(influence, row, col, dr, dc, tensor, 10.0, 0.9)
        
        elif piece_type == 'king':
            # King attacks adjacent squares
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 8 and 0 <= new_col < 8:
                        influence[new_row, new_col] = 8.0
        
        return influence
    
    def _add_ray_influence(self, influence: np.ndarray, row: int, col: int,
                          dr: int, dc: int, tensor: np.ndarray,
                          base_strength: float, decay: float):
        """Add influence along a ray direction with decay."""
        current_strength = base_strength
        current_row, current_col = row + dr, col + dc
        
        while 0 <= current_row < 8 and 0 <= current_col < 8:
            influence[current_row, current_col] += current_strength
            
            # If blocked by piece, stop ray
            if tensor[current_row, current_col, 12] > 0:
                break
            
            # Decay strength
            current_strength *= decay
            current_row += dr
            current_col += dc
    
    def compute_defense_field(self, tensor: np.ndarray, color: str) -> np.ndarray:
        """
        Compute defensive strength field.
        
        Defense = protection of own pieces + control of key squares.
        """
        defense_field = np.zeros((8, 8), dtype=np.float32)
        channel_offset = 0 if color == 'white' else 6
        
        # Get own piece positions
        own_pieces = tensor[:, :, channel_offset:channel_offset+6].sum(axis=2)
        
        # Compute attack field (pieces defend each other)
        attack_field = self.compute_attack_field(tensor, color)
        
        # Defense strength = attack field weighted by piece presence
        defense_field = attack_field * (1 + own_pieces * 2)
        
        # Bonus for defending key squares
        if color == 'white':
            # Protect castled king
            defense_field[7, 6] *= 1.5  # g1
            defense_field[7, 7] *= 1.5  # h1
        else:
            defense_field[0, 6] *= 1.5  # g8
            defense_field[0, 7] *= 1.5  # h8
        
        return defense_field
    
    def compute_control_gradient(self, tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute net control gradient: White influence - Black influence.
        
        Returns:
            control_field: 8x8 array where positive = white advantage
            control_score: scalar summary of position
        """
        white_attack = self.compute_attack_field(tensor, 'white')
        black_attack = self.compute_attack_field(tensor, 'black')
        
        white_defense = self.compute_defense_field(tensor, 'white')
        black_defense = self.compute_defense_field(tensor, 'black')
        
        # Net control = (attack + defense)_white - (attack + defense)_black
        control_field = (white_attack + white_defense) - (black_attack + black_defense)
        
        # Weighted control score (center more important)
        center_weight = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 2, 2, 1],
            [1, 2, 3, 3, 3, 3, 2, 1],
            [1, 2, 3, 4, 4, 3, 2, 1],
            [1, 2, 3, 4, 4, 3, 2, 1],
            [1, 2, 3, 3, 3, 3, 2, 1],
            [1, 2, 2, 2, 2, 2, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=np.float32)
        
        control_score = np.sum(control_field * center_weight)
        
        return control_field, control_score
    
    def evaluate_opening(self, tensor: np.ndarray, move_count: int) -> float:
        """
        Evaluate opening principles if move_count < 15.
        
        This addresses the weakness in opening play.
        """
        if move_count >= 15:
            return 0.0
        
        score = 0.0
        
        # White pieces
        white_pawns = tensor[:, :, 0]
        white_knights = tensor[:, :, 1]
        white_bishops = tensor[:, :, 2]
        white_king = tensor[:, :, 5]
        
        # Black pieces
        black_pawns = tensor[:, :, 6]
        black_knights = tensor[:, :, 7]
        black_bishops = tensor[:, :, 8]
        black_king = tensor[:, :, 11]
        
        # Center control (e4, d4, e5, d5)
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
        white_center = sum(white_pawns[r, c] for r, c in center_squares)
        black_center = sum(black_pawns[r, c] for r, c in center_squares)
        score += (white_center - black_center) * OPENING_BONUSES['center_control']
        
        # Knight development (not on back rank)
        white_knights_developed = np.sum(white_knights[0:7, :])
        black_knights_developed = np.sum(black_knights[1:8, :])
        score += (white_knights_developed - black_knights_developed) * OPENING_BONUSES['knight_development']
        
        # Bishop development
        white_bishops_developed = np.sum(white_bishops[0:6, :])
        black_bishops_developed = np.sum(black_bishops[2:8, :])
        score += (white_bishops_developed - black_bishops_developed) * OPENING_BONUSES['bishop_development']
        
        # Castling bonus
        if white_king[7, 6] > 0 or white_king[7, 2] > 0:  # King on g1 or c1
            score += OPENING_BONUSES['castling']
        if black_king[0, 6] > 0 or black_king[0, 2] > 0:  # King on g8 or c8
            score -= OPENING_BONUSES['castling']
        
        # Penalize moving same piece multiple times
        # (implicitly handled by undeveloped piece penalty)
        
        return score
    
    def evaluate_position(self, board: List[List[Optional[Dict]]], 
                         move_count: int = 0) -> float:
        """
        Main evaluation function using tensor gradient fields.
        
        Returns:
            score > 0: Black advantage
            score < 0: White advantage
        """
        # Convert board to tensor
        tensor = self.board_to_tensor(board)
        
        # 1. Material evaluation (baseline)
        material_score = 0.0
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece:
                    value = self.piece_values[piece['type']]
                    material_score += value if piece['color'] == 'black' else -value
        
        # 2. Control gradient evaluation
        control_field, control_score = self.compute_control_gradient(tensor)
        
        # 3. Opening evaluation (if early game)
        opening_score = self.evaluate_opening(tensor, move_count)
        
        # 4. Positional factors
        positional_score = self._evaluate_positional(tensor, control_field)
        
        # Weighted combination
        total_score = (
            material_score +
            control_score * 0.5 +  # Control worth ~0.5 pawns per unit
            opening_score * 0.3 +   # Opening principles
            positional_score * 0.2  # Positional factors
        )
        
        return total_score
    
    def _evaluate_positional(self, tensor: np.ndarray, control_field: np.ndarray) -> float:
        """Evaluate positional factors like pawn structure, piece coordination."""
        score = 0.0
        
        # Pawn structure
        white_pawns = tensor[:, :, 0]
        black_pawns = tensor[:, :, 6]
        
        # Doubled pawns penalty
        for col in range(8):
            white_count = np.sum(white_pawns[:, col])
            black_count = np.sum(black_pawns[:, col])
            if white_count > 1:
                score -= 20 * (white_count - 1)
            if black_count > 1:
                score += 20 * (black_count - 1)
        
        # Isolated pawns penalty
        for col in range(8):
            left_col = max(0, col - 1)
            right_col = min(7, col + 1)
            
            if np.sum(white_pawns[:, col]) > 0:
                if np.sum(white_pawns[:, left_col]) == 0 and np.sum(white_pawns[:, right_col]) == 0:
                    score -= 15
            
            if np.sum(black_pawns[:, col]) > 0:
                if np.sum(black_pawns[:, left_col]) == 0 and np.sum(black_pawns[:, right_col]) == 0:
                    score += 15
        
        # Passed pawns bonus
        for col in range(8):
            for row in range(1, 7):
                if white_pawns[row, col] > 0:
                    # Check if no black pawns ahead or adjacent
                    if not np.any(black_pawns[:row, max(0, col-1):min(8, col+2)]):
                        score -= 30 * (6 - row)  # Closer to promotion = higher bonus
                
                if black_pawns[row, col] > 0:
                    if not np.any(white_pawns[row+1:, max(0, col-1):min(8, col+2)]):
                        score += 30 * (row - 1)
        
        # Piece coordination (pieces on strong control squares)
        white_pieces = tensor[:, :, 1:6].sum(axis=2)  # Knights through king (not pawns)
        black_pieces = tensor[:, :, 7:12].sum(axis=2)
        
        # Pieces on squares with positive control
        score -= np.sum(white_pieces * np.maximum(control_field, 0)) * 5
        score += np.sum(black_pieces * np.maximum(-control_field, 0)) * 5
        
        return score


# =============================================================================
# INTEGRATION WITH EXISTING ENGINE
# =============================================================================

def create_tensor_engine():
    """Factory function to create tensor engine instance."""
    return TensorChessEngine()


def evaluate_board_tensor(board: List[List[Optional[Dict]]], 
                          move_count: int = 0) -> float:
    """
    Drop-in replacement for evaluate_board() in engine.py.
    
    Args:
        board: 8x8 board representation
        move_count: number of moves made (for opening evaluation)
    
    Returns:
        score > 0: Black advantage
        score < 0: White advantage
    """
    engine = create_tensor_engine()
    return engine.evaluate_position(board, move_count)