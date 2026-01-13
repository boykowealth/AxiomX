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

def evaluate_board(board):
    piece_values = {
        'pawn': 100,
        'knight': 320,
        'bishop': 330,
        'rook': 500,
        'queen': 900,
        'king': 20000
    }
    
    pst_pawn = [
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [ 50, 50, 50, 50, 50, 50, 50, 50],
        [ 10, 10, 20, 30, 30, 20, 10, 10],
        [  5,  5, 10, 25, 25, 10,  5,  5],
        [  0,  0,  0, 20, 20,  0,  0,  0],
        [  5, -5,-10,  0,  0,-10, -5,  5],
        [  5, 10, 10,-20,-20, 10, 10,  5],
        [  0,  0,  0,  0,  0,  0,  0,  0]
    ]
    
    pst_knight = [
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,  0,  5,  5,  0,-20,-40],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-30,  0, 15, 20, 20, 15,  0,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  0, 10, 15, 15, 10,  0,-30],
        [-40,-20,  0,  0,  0,  0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50]
    ]
    
    pst_bishop = [
        [-20,-10,-10,-10,-10,-10,-10,-20],
        [-10,  5,  0,  0,  0,  0,  5,-10],
        [-10, 10, 10, 10, 10, 10, 10,-10],
        [-10,  0, 10, 10, 10, 10,  0,-10],
        [-10,  5,  5, 10, 10,  5,  5,-10],
        [-10,  0,  5, 10, 10,  5,  0,-10],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-20,-10,-10,-10,-10,-10,-10,-20]
    ]
    
    pst_king = [
        [ 20, 30, 10,  0,  0, 10, 30, 20],
        [ 20, 20,  0,  0,  0,  0, 20, 20],
        [-10,-20,-20,-20,-20,-20,-20,-10],
        [-20,-30,-30,-40,-40,-30,-30,-20],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30]
    ]
    
    score = 0
    
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece:
                value = piece_values[piece['type']]
                pos_row = row if piece['color'] == 'black' else 7 - row
                
                if piece['type'] == 'pawn':
                    value += pst_pawn[pos_row][col]
                elif piece['type'] == 'knight':
                    value += pst_knight[pos_row][col]
                elif piece['type'] == 'bishop':
                    value += pst_bishop[pos_row][col]
                elif piece['type'] == 'king':
                    value += pst_king[pos_row][col]
                
                score += value if piece['color'] == 'black' else -value
    
    return score

def score_move(board, move):
    from_row, from_col, to_row, to_col = move
    target = board[to_row][to_col]
    
    if target:
        piece_values = {'pawn': 1, 'knight': 3, 'bishop': 3, 'rook': 5, 'queen': 9, 'king': 0}
        piece = board[from_row][from_col]
        return 10 + piece_values.get(target['type'], 0) - piece_values.get(piece['type'], 0)
    
    return 0

def order_moves(board, moves):
    captures = []
    non_captures = []
    for move in moves:
        if board[move[2]][move[3]]:
            captures.append((score_move(board, move), move))
        else:
            non_captures.append(move)
    
    captures.sort(reverse=True, key=lambda x: x[0])
    return [move for _, move in captures] + non_captures

def minimax(board, depth, alpha, beta, maximizing, en_passant=None):
    if depth == 0:
        return evaluate_board(board), None
    
    color = 'black' if maximizing else 'white'
    moves = get_all_valid_moves(board, color, en_passant)
    
    if not moves:
        if is_in_check(board, color):
            return (-100000 if maximizing else 100000), None
        return 0, None
    
    moves = order_moves(board, moves)
    best_move = moves[0]
    
    if maximizing:
        max_eval = float('-inf')
        for move in moves:
            temp_board = copy_board(board)
            if en_passant and (move[2], move[3]) == en_passant:
                temp_board[move[0]][move[3]] = None
            temp_board[move[2]][move[3]] = temp_board[move[0]][move[1]]
            temp_board[move[0]][move[1]] = None
            eval_score, _ = minimax(temp_board, depth - 1, alpha, beta, False, None)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in moves:
            temp_board = copy_board(board)
            if en_passant and (move[2], move[3]) == en_passant:
                temp_board[move[0]][move[3]] = None
            temp_board[move[2]][move[3]] = temp_board[move[0]][move[1]]
            temp_board[move[0]][move[1]] = None
            eval_score, _ = minimax(temp_board, depth - 1, alpha, beta, True, None)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

def get_ai_move(board, en_passant=None, color='black'):
    maximizing = (color == 'black')
    _, best_move = minimax(board, 3, float('-inf'), float('inf'), maximizing, en_passant)
    return best_move