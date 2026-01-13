from dash import html
import dash_bootstrap_components as dbc

PIECES = {
    'white': {
        'king': '♚',
        'queen': '♛',
        'rook': '♜',
        'bishop': '♝',
        'knight': '♞',
        'pawn': '♙'
    },
    'black': {
        'king': '♚',
        'queen': '♛',
        'rook': '♜',
        'bishop': '♝',
        'knight': '♞',
        'pawn': '♙'
    }
}

def position_to_algebraic(row, col):
    files = 'abcdefgh'
    ranks = '87654321'
    return f"{files[col]}{ranks[row]}"

def move_to_algebraic(board, from_row, from_col, to_row, to_col):
    piece = board[from_row][from_col]
    if not piece:
        return ""
    
    piece_symbol = ''
    if piece['type'] != 'pawn':
        piece_symbol = piece['type'][0].upper()
        if piece['type'] == 'knight':
            piece_symbol = 'N'
    
    from_pos = position_to_algebraic(from_row, from_col)
    to_pos = position_to_algebraic(to_row, to_col)
    
    target = board[to_row][to_col]
    capture = 'x' if target else ''
    
    if piece['type'] == 'pawn' and capture:
        return f"{from_pos[0]}{capture}{to_pos}"
    
    return f"{piece_symbol}{capture}{to_pos}"

def create_board_layout(board, selected=None, valid_moves=None):
    if valid_moves is None:
        valid_moves = []
    squares = []
    
    for row in range(8):
        for col in range(8):
            is_light = (row + col) % 2 == 0
            bg_color = '#333333' if is_light else '#000000'
            
            if selected and selected == (row, col):
                bg_color = '#00ff00'
            
            piece = board[row][col]
            piece_symbol = ''
            piece_color = '#000000'
            
            if piece:
                piece_symbol = PIECES[piece['color']][piece['type']]
                piece_color = '#ffffff' if piece['color'] == 'white' else '#00ff00'
            
            square_children = [piece_symbol]
            
            if (row, col) in valid_moves:
                square_children.append(
                    html.Div(style={
                        'position': 'absolute',
                        'width': '20%',
                        'height': '20%',
                        'backgroundColor': '#666666',
                        'borderRadius': '50%',
                        'pointerEvents': 'none'
                    })
                )
            
            square = html.Div(
                square_children,
                id={'type': 'square', 'row': row, 'col': col},
                style={
                    'width': '100%',
                    'height': '100%',
                    'backgroundColor': bg_color,
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'fontSize': '3vw',
                    'cursor': 'pointer',
                    'userSelect': 'none',
                    'color': piece_color,
                    'fontWeight': 'bold',
                    'position': 'relative'
                }
            )
            squares.append(square)
    
    return html.Div(
        squares,
        style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(8, 1fr)',
            'gridTemplateRows': 'repeat(8, 1fr)',
            'width': '100%',
            'height': '100%',
            'border': '2px solid #ffffff'
        }
    )

def create_evaluation_bar(evaluation):
    clamped_eval = max(-10, min(10, evaluation / 100))
    white_percentage = 50 - (clamped_eval * 5)
    black_percentage = 50 + (clamped_eval * 5)
    
    black_score = f"+{abs(evaluation/100):.1f}" if evaluation > 0 else ""
    white_score = f"+{abs(evaluation/100):.1f}" if evaluation < 0 else ""
    
    return html.Div([
        html.Div(
            black_score,
            style={
                'fontSize': '10px',
                'color': '#00ff00',
                'textAlign': 'center',
                'padding': '2px 0',
                'fontFamily': 'Courier New, monospace',
                'fontWeight': 'bold'
            }
        ),
        html.Div([
            html.Div(
                style={
                    'height': f'{black_percentage}%',
                    'backgroundColor': '#00ff00',
                    'transition': 'height 0.5s ease'
                }
            ),
            html.Div(
                style={
                    'height': f'{white_percentage}%',
                    'backgroundColor': '#ffffff',
                    'transition': 'height 0.5s ease'
                }
            )
        ], style={
            'flex': '1',
            'display': 'flex',
            'flexDirection': 'column',
            'border': '1px solid #ffffff',
            'overflow': 'hidden'
        }),
        html.Div(
            white_score,
            style={
                'fontSize': '10px',
                'color': '#ffffff',
                'textAlign': 'center',
                'padding': '2px 0',
                'fontFamily': 'Courier New, monospace',
                'fontWeight': 'bold'
            }
        )
    ], style={
        'height': '100%',
        'width': '100%',
        'display': 'flex',
        'flexDirection': 'column'
    })

def create_move_list(moves):
    if not moves:
        return html.Div(
            "> No moves yet",
            style={
                'color': '#ffffff',
                'fontFamily': 'Courier New, monospace',
                'fontSize': '14px',
                'padding': '10px'
            }
        )
    
    move_pairs = []
    for i in range(0, len(moves), 2):
        move_num = (i // 2) + 1
        white_move = moves[i]
        black_move = moves[i + 1] if i + 1 < len(moves) else ''
        
        move_pairs.append(
            html.Div([
                html.Span(f"{move_num}.", style={
                    'minWidth': '35px',
                    'display': 'inline-block',
                    'color': '#ffffff'
                }),
                html.Span(white_move, style={
                    'minWidth': '70px',
                    'display': 'inline-block',
                    'marginRight': '15px',
                    'color': '#ffffff'
                }),
                html.Span(black_move, style={
                    'minWidth': '70px',
                    'display': 'inline-block',
                    'color': '#00ff00'
                })
            ], style={
                'fontFamily': 'Courier New, monospace',
                'fontSize': '14px',
                'marginBottom': '4px',
                'padding': '2px 0'
            })
        )
    
    return html.Div(
        move_pairs,
        style={
            'padding': '10px'
        }
    )

def create_promotion_modal():
    return dbc.Modal([
        dbc.ModalHeader(
            "PROMOTE PAWN",
            style={
                'backgroundColor': '#000000',
                'color': '#ffffff',
                'fontFamily': 'Courier New, monospace',
                'border': '1px solid #ffffff'
            }
        ),
        dbc.ModalBody([
            dbc.Button(
                "Queen ♕",
                id='promote-queen',
                style={
                    'backgroundColor': '#000000',
                    'color': '#ffffff',
                    'border': '1px solid #ffffff',
                    'fontFamily': 'Courier New, monospace',
                    'margin': '5px'
                },
                className='m-2',
                size='lg'
            ),
            dbc.Button(
                "Rook ♖",
                id='promote-rook',
                style={
                    'backgroundColor': '#000000',
                    'color': '#ffffff',
                    'border': '1px solid #ffffff',
                    'fontFamily': 'Courier New, monospace',
                    'margin': '5px'
                },
                className='m-2',
                size='lg'
            ),
            dbc.Button(
                "Bishop ♗",
                id='promote-bishop',
                style={
                    'backgroundColor': '#000000',
                    'color': '#ffffff',
                    'border': '1px solid #ffffff',
                    'fontFamily': 'Courier New, monospace',
                    'margin': '5px'
                },
                className='m-2',
                size='lg'
            ),
            dbc.Button(
                "Knight ♘",
                id='promote-knight',
                style={
                    'backgroundColor': '#000000',
                    'color': '#ffffff',
                    'border': '1px solid #ffffff',
                    'fontFamily': 'Courier New, monospace',
                    'margin': '5px'
                },
                className='m-2',
                size='lg'
            ),
        ], style={
            'backgroundColor': '#000000',
            'border': '1px solid #ffffff'
        })
    ], id='promotion-modal', is_open=False, centered=True, style={
        'fontFamily': 'Courier New, monospace'
    })

def create_gradient_visualization(board):
    """
    Create terminal-style gradient visualization.
    
    Computes control gradient and displays as ASCII-art grid.
    Green = White advantage, Red = Black advantage, Grey = Neutral
    """
    try:
        from tensor_engine import TensorChessEngine
        
        engine = TensorChessEngine()
        tensor = engine.board_to_tensor(board)
        control_field, _ = engine.compute_control_gradient(tensor)
        
        squares = []
        for row in range(8):
            for col in range(8):
                value = control_field[row, col]
                
                normalized = max(-1, min(1, value / 50.0))
                
                if normalized > 0.3:
                    intensity = int(min(255, 100 + normalized * 155))
                    bg_color = f'rgb(0, {intensity}, 0)'
                    symbol = '+'
                elif normalized < -0.3:
                    intensity = int(min(255, 100 + abs(normalized) * 155))
                    bg_color = f'rgb({intensity}, 0, 0)'
                    symbol = '-'
                else:
                    bg_color = '#333333'
                    symbol = '·'
                
                square = html.Div(
                    symbol,
                    style={
                        'width': '100%',
                        'height': '100%',
                        'backgroundColor': bg_color,
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'fontSize': '10px',
                        'color': '#ffffff',
                        'fontFamily': 'Courier New, monospace',
                        'fontWeight': 'bold',
                        'border': '1px solid #000000'
                    }
                )
                squares.append(square)
        
        grid = html.Div(
            squares,
            style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(8, 1fr)',
                'gridTemplateRows': 'repeat(8, 1fr)',
                'width': '100%',
                'aspectRatio': '1',
                'border': '1px solid #ffffff',
                'maxWidth': '200px'
            }
        )
        
        legend = html.Div([
            html.Div([
                html.Span('+ ', style={'color': '#00ff00', 'marginRight': '5px'}),
                html.Span('WHITE ADVANTAGE', style={'fontSize': '10px'})
            ], style={'marginBottom': '3px'}),
            html.Div([
                html.Span('· ', style={'color': '#ffffff', 'marginRight': '5px'}),
                html.Span('NEUTRAL', style={'fontSize': '10px'})
            ], style={'marginBottom': '3px'}),
            html.Div([
                html.Span('- ', style={'color': '#ff0000', 'marginRight': '5px'}),
                html.Span('BLACK ADVANTAGE', style={'fontSize': '10px'})
            ])
        ], style={
            'color': '#ffffff',
            'fontFamily': 'Courier New, monospace',
            'fontSize': '10px',
            'marginTop': '10px'
        })
        
        return html.Div([grid, legend])
        
    except Exception as e:
        return html.Div(
            f"[Gradient unavailable]",
            style={
                'color': '#666666',
                'fontFamily': 'Courier New, monospace',
                'fontSize': '12px',
                'fontStyle': 'italic'
            }
        )