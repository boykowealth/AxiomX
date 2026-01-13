import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import json

from engine import initialize_board, is_in_check, evaluate_board
from ui import (
    create_board_layout, create_evaluation_bar, 
    create_move_list, create_promotion_modal, create_gradient_visualization
)
from logic import handle_player_move, handle_ai_move, handle_promotion

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .form-check-input {
                background-color: #000000 !important;
                border-color: #ffffff !important;
            }
            .form-check-input:checked {
                background-color: #00ff00 !important;
                border-color: #00ff00 !important;
            }
            .form-check-label {
                color: #ffffff !important;
                font-family: 'Courier New', monospace !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div(
                id='board-container',
                style={
                    'width': '90%',
                    'paddingBottom': '90%',
                    'position': 'relative',
                    'margin': '0 auto'
                }
            ),
            html.Div(
                id='game-status',
                className='mt-3 text-center',
                style={
                    'fontSize': '16px',
                    'color': '#ffffff',
                    'fontFamily': 'Courier New, monospace'
                }
            )
        ], style={
            'width': '50%',
            'height': '100vh',
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'justifyContent': 'center',
            'padding': '20px',
            'backgroundColor': '#000000'
        }),
        
        html.Div([
            html.Div(
                id='evaluation-bar',
                style={
                    'width': '50px',
                    'height': '100vh',
                    'padding': '10px 5px'
                }
            ),
            
            html.Div([
                html.Div([
                    html.Div(
                        "MOVE HISTORY",
                        style={
                            'color': '#ffffff',
                            'fontFamily': 'Courier New, monospace',
                            'fontSize': '14px',
                            'fontWeight': 'bold',
                            'marginBottom': '10px',
                            'borderBottom': '1px solid #ffffff',
                            'paddingBottom': '5px'
                        }
                    ),
                    html.Div(
                        id='move-list-container',
                        style={
                            'height': '200px',
                            'overflowY': 'auto',
                            'marginBottom': '15px'
                        }
                    )
                ], style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'marginBottom': '15px'
                }),
                
                html.Div([
                    html.Div(
                        "POSITION GRADIENT",
                        style={
                            'color': '#ffffff',
                            'fontFamily': 'Courier New, monospace',
                            'fontSize': '14px',
                            'fontWeight': 'bold',
                            'marginBottom': '10px',
                            'borderBottom': '1px solid #ffffff',
                            'paddingBottom': '5px'
                        }
                    ),
                    html.Div(id='gradient-visualization', style={'marginBottom': '15px'})
                ], style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'marginBottom': '15px'
                }),
                
                html.Div([
                    dbc.Button(
                        "NEW GAME",
                        id='reset-button',
                        size='sm',
                        style={
                            'backgroundColor': '#000000',
                            'color': '#ffffff',
                            'border': '1px solid #ffffff',
                            'fontFamily': 'Courier New, monospace',
                            'fontWeight': 'bold',
                            'width': '100%',
                            'marginBottom': '10px'
                        }
                    ),
                    html.Div([
                        html.Span("PLAY AS: ", style={
                            'color': '#ffffff',
                            'fontFamily': 'Courier New, monospace',
                            'fontSize': '12px',
                            'marginRight': '10px'
                        }),
                        dbc.RadioItems(
                            id='player-color',
                            options=[
                                {'label': 'WHITE', 'value': 'white'},
                                {'label': 'BLACK', 'value': 'black'}
                            ],
                            value='white',
                            inline=True,
                            style={
                                'color': '#ffffff',
                                'fontFamily': 'Courier New, monospace',
                                'fontSize': '12px'
                            }
                        )
                    ], style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'padding': '10px 0'
                    })
                ], style={
                    'borderTop': '1px solid #ffffff',
                    'paddingTop': '15px'
                })
            ], style={
                'flex': '1',
                'height': '100vh',
                'padding': '20px',
                'display': 'flex',
                'flexDirection': 'column'
            })
        ], style={
            'width': '50%',
            'height': '100vh',
            'backgroundColor': '#000000',
            'borderLeft': '1px solid #ffffff',
            'display': 'flex',
            'flexDirection': 'row'
        })
    ], style={
        'display': 'flex',
        'width': '100%',
        'height': '100vh',
        'backgroundColor': '#000000'
    }),
    
    create_promotion_modal(),
    
    dcc.Store(id='game-state', data={
        'board': initialize_board(),
        'current_player': 'white',
        'player_color': 'white',
        'selected': None,
        'valid_moves': [],
        'game_over': None,
        'en_passant': None,
        'pending_promotion': None,
        'move_history': [],
        'evaluation': 0,
        'move_count': 0,  # Track move count for opening eval
        'ai_thinking': False  # NEW: Track if AI should calculate
    }),
    
    # Interval component for deferred AI calculation
    dcc.Interval(
        id='ai-move-interval',
        interval=50,  # Check every 50ms
        n_intervals=0,
        disabled=True  # Start disabled
    )
], style={'overflow': 'hidden'})

@app.callback(
    [Output('board-container', 'children'),
     Output('game-state', 'data'),
     Output('game-status', 'children'),
     Output('promotion-modal', 'is_open'),
     Output('evaluation-bar', 'children'),
     Output('move-list-container', 'children'),
     Output('gradient-visualization', 'children'),  # NEW: Gradient viz
     Output('ai-move-interval', 'disabled')],  # Control interval
    [Input({'type': 'square', 'row': dash.dependencies.ALL, 'col': dash.dependencies.ALL}, 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('player-color', 'value'),
     Input('promote-queen', 'n_clicks'),
     Input('promote-rook', 'n_clicks'),
     Input('promote-bishop', 'n_clicks'),
     Input('promote-knight', 'n_clicks'),
     Input('ai-move-interval', 'n_intervals')],  # NEW: Trigger for AI calculation
    [State('game-state', 'data')],
    prevent_initial_call=False
)
def handle_click(square_clicks, reset_clicks, player_color, promote_queen, promote_rook, 
                 promote_bishop, promote_knight, ai_interval, game_state):
    ctx = callback_context
    
    # Ensure move_count exists
    if 'move_count' not in game_state:
        game_state['move_count'] = 0
    
    # Ensure ai_thinking flag exists
    if 'ai_thinking' not in game_state:
        game_state['ai_thinking'] = False
    
    # Handle AI calculation via interval
    if 'ai-move-interval' in ctx.triggered[0]['prop_id'] if ctx.triggered else False:
        if game_state.get('ai_thinking', False):
            # AI's turn - calculate move
            handle_ai_move(game_state)
            game_state['ai_thinking'] = False
            
            board_layout_wrapper = html.Div(
                create_board_layout(game_state['board']),
                style={'position': 'absolute', 'top': 0, 'left': 0, 'right': 0, 'bottom': 0}
            )
            status = game_state['game_over'] if game_state['game_over'] else f"> {game_state['current_player'].upper()}'S TURN"
            if not game_state['game_over'] and is_in_check(game_state['board'], game_state['current_player']):
                status += " [CHECK!]"
            eval_bar = create_evaluation_bar(game_state['evaluation'])
            move_list = create_move_list(game_state['move_history'])
            gradient_viz = create_gradient_visualization(game_state['board'])
            return board_layout_wrapper, game_state, status, False, eval_bar, move_list, gradient_viz, True  # Disable interval
    
    if player_color and player_color != game_state.get('player_color', 'white'):
        game_state['player_color'] = player_color
        if player_color == 'black' and game_state['current_player'] == 'white' and not game_state.get('game_over'):
            # Set flag for AI to think, enable interval
            game_state['ai_thinking'] = True
            board_layout_wrapper = html.Div(
                create_board_layout(game_state['board']),
                style={'position': 'absolute', 'top': 0, 'left': 0, 'right': 0, 'bottom': 0}
            )
            status = "> AI THINKING..."
            eval_bar = create_evaluation_bar(game_state['evaluation'])
            move_list = create_move_list(game_state['move_history'])
            gradient_viz = create_gradient_visualization(game_state['board'])
            return board_layout_wrapper, game_state, status, False, eval_bar, move_list, gradient_viz, False  # Enable interval
    
    if not ctx.triggered:
        board_layout_wrapper = html.Div(
            create_board_layout(game_state['board']),
            style={
                'position': 'absolute',
                'top': 0,
                'left': 0,
                'right': 0,
                'bottom': 0
            }
        )
        status = f"> {game_state['current_player'].upper()}'S TURN"
        eval_bar = create_evaluation_bar(game_state['evaluation'])
        move_list = create_move_list(game_state['move_history'])
        gradient_viz = create_gradient_visualization(game_state['board'])
        return board_layout_wrapper, game_state, status, False, eval_bar, move_list, gradient_viz, True  # Disable interval
    
    trigger_id = ctx.triggered[0]['prop_id']
    
    if any(x in trigger_id for x in ['promote-queen', 'promote-rook', 'promote-bishop', 'promote-knight']):
        if game_state['pending_promotion']:
            promotion_map = {
                'promote-queen': 'queen',
                'promote-rook': 'rook',
                'promote-bishop': 'bishop',
                'promote-knight': 'knight'
            }
            for key, piece_type in promotion_map.items():
                if key in trigger_id:
                    handle_promotion(game_state, piece_type)
                    # Set flag for AI to think
                    game_state['ai_thinking'] = True
                    break
            
            board_layout_wrapper = html.Div(
                create_board_layout(game_state['board']),
                style={'position': 'absolute', 'top': 0, 'left': 0, 'right': 0, 'bottom': 0}
            )
            status = "> AI THINKING..."
            eval_bar = create_evaluation_bar(game_state['evaluation'])
            move_list = create_move_list(game_state['move_history'])
            gradient_viz = create_gradient_visualization(game_state['board'])
            return board_layout_wrapper, game_state, status, False, eval_bar, move_list, gradient_viz, False  # Enable interval
    
    if 'reset-button' in trigger_id:
        player_color = game_state.get('player_color', 'white')
        new_state = {
            'board': initialize_board(),
            'current_player': 'white',
            'player_color': player_color,
            'selected': None,
            'valid_moves': [],
            'game_over': None,
            'en_passant': None,
            'pending_promotion': None,
            'move_history': [],
            'evaluation': 0,
            'move_count': 0,
            'ai_thinking': False
        }
        
        board_layout_wrapper = html.Div(
            create_board_layout(new_state['board']),
            style={'position': 'absolute', 'top': 0, 'left': 0, 'right': 0, 'bottom': 0}
        )
        
        if player_color == 'black':
            # AI starts, enable interval
            new_state['ai_thinking'] = True
            status = "> AI THINKING..."
            eval_bar = create_evaluation_bar(new_state['evaluation'])
            move_list = create_move_list(new_state['move_history'])
            gradient_viz = create_gradient_visualization(new_state['board'])
            return board_layout_wrapper, new_state, status, False, eval_bar, move_list, gradient_viz, False  # Enable interval
        else:
            status = f"> {new_state['current_player'].upper()}'S TURN"
            eval_bar = create_evaluation_bar(new_state['evaluation'])
            move_list = create_move_list(new_state['move_history'])
            gradient_viz = create_gradient_visualization(new_state['board'])
            return board_layout_wrapper, new_state, status, False, eval_bar, move_list, gradient_viz, True  # Disable interval
    
    player_color = game_state.get('player_color', 'white')
    ai_color = 'black' if player_color == 'white' else 'white'
    
    if game_state['game_over'] or game_state['current_player'] == ai_color or game_state['pending_promotion']:
        board_layout_wrapper = html.Div(
            create_board_layout(
                game_state['board'],
                tuple(game_state['selected']) if game_state['selected'] else None,
                [tuple(m) for m in game_state['valid_moves']]
            ),
            style={'position': 'absolute', 'top': 0, 'left': 0, 'right': 0, 'bottom': 0}
        )
        status = game_state['game_over'] if game_state['game_over'] else f"> {game_state['current_player'].upper()}'S TURN"
        eval_bar = create_evaluation_bar(game_state['evaluation'])
        move_list = create_move_list(game_state['move_history'])
        gradient_viz = create_gradient_visualization(game_state['board'])
        return board_layout_wrapper, game_state, status, game_state['pending_promotion'] is not None, eval_bar, move_list, gradient_viz, True  # Disable interval
    
    # Only parse JSON if this is a square click (not ai-interval or other triggers)
    if 'square' not in trigger_id:
        # Not a square click, just refresh the display
        board_layout_wrapper = html.Div(
            create_board_layout(
                game_state['board'],
                tuple(game_state['selected']) if game_state['selected'] else None,
                [tuple(m) for m in game_state['valid_moves']]
            ),
            style={'position': 'absolute', 'top': 0, 'left': 0, 'right': 0, 'bottom': 0}
        )
        status = game_state['game_over'] if game_state['game_over'] else f"> {game_state['current_player'].upper()}'S TURN"
        if not game_state['game_over'] and is_in_check(game_state['board'], game_state['current_player']):
            status += " [CHECK!]"
        eval_bar = create_evaluation_bar(game_state['evaluation'])
        move_list = create_move_list(game_state['move_history'])
        gradient_viz = create_gradient_visualization(game_state['board'])
        return board_layout_wrapper, game_state, status, False, eval_bar, move_list, gradient_viz, True
    
    trigger_dict = json.loads(trigger_id.split('.')[0])
    clicked_row, clicked_col = trigger_dict['row'], trigger_dict['col']
    
    needs_promotion = handle_player_move(game_state, clicked_row, clicked_col)
    
    if needs_promotion:
        board_layout_wrapper = html.Div(
            create_board_layout(game_state['board']),
            style={'position': 'absolute', 'top': 0, 'left': 0, 'right': 0, 'bottom': 0}
        )
        eval_bar = create_evaluation_bar(game_state['evaluation'])
        move_list = create_move_list(game_state['move_history'])
        gradient_viz = create_gradient_visualization(game_state['board'])
        return board_layout_wrapper, game_state, "> SELECT PROMOTION", True, eval_bar, move_list, gradient_viz, True  # Disable interval
    
    # Check if it's now AI's turn
    if game_state['current_player'] == ai_color:
        # Set flag and enable interval for AI to calculate in background
        game_state['ai_thinking'] = True
        
        # Immediately show player's move with "AI THINKING" status
        board_layout_wrapper = html.Div(
            create_board_layout(game_state['board']),
            style={'position': 'absolute', 'top': 0, 'left': 0, 'right': 0, 'bottom': 0}
        )
        status = "> AI THINKING..."
        eval_bar = create_evaluation_bar(game_state['evaluation'])
        move_list = create_move_list(game_state['move_history'])
        gradient_viz = create_gradient_visualization(game_state['board'])
        return board_layout_wrapper, game_state, status, False, eval_bar, move_list, gradient_viz, False  # Enable interval for AI
    
    # Player's turn still
    board_layout_wrapper = html.Div(
        create_board_layout(
            game_state['board'],
            tuple(game_state['selected']) if game_state['selected'] else None,
            [tuple(m) for m in game_state['valid_moves']]
        ),
        style={'position': 'absolute', 'top': 0, 'left': 0, 'right': 0, 'bottom': 0}
    )
    status = game_state['game_over'] if game_state['game_over'] else f"> {game_state['current_player'].upper()}'S TURN"
    if not game_state['game_over'] and is_in_check(game_state['board'], game_state['current_player']):
        status += " [CHECK!]"
    
    eval_bar = create_evaluation_bar(game_state['evaluation'])
    move_list = create_move_list(game_state['move_history'])
    gradient_viz = create_gradient_visualization(game_state['board'])
    
    return board_layout_wrapper, game_state, status, False, eval_bar, move_list, gradient_viz, True  # Disable interval

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)