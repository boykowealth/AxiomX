from engine import (
    copy_board, is_valid_move, would_be_in_check, 
    get_ai_move, get_all_valid_moves, is_in_check, evaluate_board
)
from ui import move_to_algebraic
import time


def execute_move(board, from_row, from_col, to_row, to_col, en_passant=None):
    """Execute a move on the board."""
    piece = board[from_row][from_col]
    
    # Handle en passant capture
    if piece['type'] == 'pawn' and en_passant and (to_row, to_col) == en_passant:
        board[from_row][to_col] = None
    
    # Handle castling
    if piece['type'] == 'king' and abs(to_col - from_col) == 2:
        if to_col == 6:  # Kingside
            board[from_row][5] = board[from_row][7]
            board[from_row][7] = None
            board[from_row][5]['moved'] = True
        elif to_col == 2:  # Queenside
            board[from_row][3] = board[from_row][0]
            board[from_row][0] = None
            board[from_row][3]['moved'] = True
    
    # Track en passant target square
    new_en_passant = None
    if piece['type'] == 'pawn' and abs(to_row - from_row) == 2:
        ep_row = (from_row + to_row) // 2
        new_en_passant = (ep_row, to_col)
    
    # Execute move
    board[to_row][to_col] = piece
    board[from_row][from_col] = None
    piece['moved'] = True
    
    return new_en_passant


def handle_player_move(game_state, clicked_row, clicked_col):
    """Handle player move with move count tracking."""
    board = game_state['board']
    selected = tuple(game_state['selected']) if game_state['selected'] else None
    player_color = game_state.get('player_color', 'white')
    
    # Ensure move_count exists in game state
    if 'move_count' not in game_state:
        game_state['move_count'] = 0
    
    if selected is None:
        # Select piece
        piece = board[clicked_row][clicked_col]
        if piece and piece['color'] == player_color:
            valid_moves = []
            for to_row in range(8):
                for to_col in range(8):
                    if is_valid_move(board, clicked_row, clicked_col, to_row, to_col, 
                                   tuple(game_state['en_passant']) if game_state['en_passant'] else None):
                        if not would_be_in_check(board, clicked_row, clicked_col, to_row, to_col, 
                                                player_color,
                                                tuple(game_state['en_passant']) if game_state['en_passant'] else None):
                            valid_moves.append((to_row, to_col))
            game_state['selected'] = [clicked_row, clicked_col]
            game_state['valid_moves'] = valid_moves
    else:
        # Execute move if valid
        if (clicked_row, clicked_col) in [tuple(m) for m in game_state['valid_moves']]:
            piece = board[selected[0]][selected[1]]
            
            move_notation = move_to_algebraic(board, selected[0], selected[1], clicked_row, clicked_col)
            
            new_en_passant = execute_move(
                board, selected[0], selected[1], clicked_row, clicked_col,
                tuple(game_state['en_passant']) if game_state['en_passant'] else None
            )
            game_state['en_passant'] = list(new_en_passant) if new_en_passant else None
            
            game_state['selected'] = None
            game_state['valid_moves'] = []
            
            # Increment move count
            game_state['move_count'] += 1
            
            # Handle pawn promotion
            if piece['type'] == 'pawn' and clicked_row == 0:
                game_state['pending_promotion'] = [clicked_row, clicked_col]
                game_state['move_history'].append(move_notation)
                return True
            
            game_state['move_history'].append(move_notation)
            
            # Evaluate with move count for opening awareness
            game_state['evaluation'] = evaluate_board(board, game_state['move_count'])
            
            # Switch turn
            game_state['current_player'] = 'black' if game_state['current_player'] == 'white' else 'white'
            return False
        else:
            # Reselect different piece
            piece = board[clicked_row][clicked_col]
            if piece and piece['color'] == player_color:
                valid_moves = []
                for to_row in range(8):
                    for to_col in range(8):
                        if is_valid_move(board, clicked_row, clicked_col, to_row, to_col, 
                                       tuple(game_state['en_passant']) if game_state['en_passant'] else None):
                            if not would_be_in_check(board, clicked_row, clicked_col, to_row, to_col, 
                                                    player_color,
                                                    tuple(game_state['en_passant']) if game_state['en_passant'] else None):
                                valid_moves.append((to_row, to_col))
                game_state['selected'] = [clicked_row, clicked_col]
                game_state['valid_moves'] = valid_moves
            else:
                game_state['selected'] = None
                game_state['valid_moves'] = []
    
    return False


def handle_ai_move(game_state):
    """Handle AI move with move count tracking."""
    board = game_state['board']
    
    # Ensure move_count exists
    if 'move_count' not in game_state:
        game_state['move_count'] = 0
    
    ai_color = game_state['current_player']
    ai_move = get_ai_move(
        board, 
        tuple(game_state['en_passant']) if game_state['en_passant'] else None, 
        ai_color,
        game_state['move_count']
    )
    
    if ai_move:
        move_notation = move_to_algebraic(board, ai_move[0], ai_move[1], ai_move[2], ai_move[3])
        
        new_en_passant = execute_move(
            board, ai_move[0], ai_move[1], ai_move[2], ai_move[3],
            tuple(game_state['en_passant']) if game_state['en_passant'] else None
        )
        game_state['en_passant'] = list(new_en_passant) if new_en_passant else None
        
        # Increment move count
        game_state['move_count'] += 1
        
        # Auto-promote to queen
        ai_piece = board[ai_move[2]][ai_move[3]]
        if ai_piece['type'] == 'pawn' and ai_move[2] == 7:
            board[ai_move[2]][ai_move[3]]['type'] = 'queen'
            move_notation += '=Q'
        
        game_state['move_history'].append(move_notation)
        
        # Evaluate with move count
        game_state['evaluation'] = evaluate_board(board, game_state['move_count'])
        
        # Check for game over
        opponent_color = 'white' if ai_color == 'black' else 'black'
        game_state['current_player'] = opponent_color
        
        if not get_all_valid_moves(board, opponent_color, tuple(game_state['en_passant']) if game_state['en_passant'] else None):
            if is_in_check(board, opponent_color):
                winner = 'Black' if ai_color == 'black' else 'White'
                game_state['game_over'] = f"Checkmate! {winner} wins!"
            else:
                game_state['game_over'] = "Stalemate!"
    else:
        # AI has no moves
        if is_in_check(board, ai_color):
            winner = 'White' if ai_color == 'black' else 'Black'
            game_state['game_over'] = f"Checkmate! {winner} wins!"
        else:
            game_state['game_over'] = "Stalemate!"


def handle_promotion(game_state, piece_type):
    """Handle pawn promotion."""
    if game_state['pending_promotion']:
        row, col = game_state['pending_promotion']
        board = game_state['board']
        board[row][col]['type'] = piece_type
        
        # Update move notation
        if game_state['move_history']:
            game_state['move_history'][-1] += f'={piece_type[0].upper()}'
        
        game_state['pending_promotion'] = None
        
        # Re-evaluate with move count
        if 'move_count' not in game_state:
            game_state['move_count'] = 0
        game_state['evaluation'] = evaluate_board(board, game_state['move_count'])
        
        # Switch turn
        game_state['current_player'] = 'black' if game_state['current_player'] == 'white' else 'white'