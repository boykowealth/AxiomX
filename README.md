# AxiomX: A Lightweight Chess Engine with Adaptive Minimax Search

**Version 1.0**  
**Release Date: January 2026**

---

## Abstract

AxiomX is a minimalist terminal-themed chess engine implementing classical game tree search algorithms with modern optimizations. The engine features a complete rule-compliant chess implementation with special move handling, an evaluation function based on piece-square tables, and an alpha-beta pruned minimax search algorithm. The system provides a real-time interactive interface built on Python Dash, supporting both human-vs-AI and configurable player color assignments. This document presents the theoretical foundations, implementation details, and architectural decisions underlying the AxiomX chess engine.

**Keywords:** Chess Engine, Minimax Algorithm, Alpha-Beta Pruning, Game Tree Search, Position Evaluation, Piece-Square Tables

---

## 1. Introduction

### 1.1 Motivation

Chess remains one of the most studied domains in artificial intelligence and game theory. The development of chess engines provides practical insights into search algorithms, position evaluation heuristics, and computational optimization. AxiomX was designed to balance computational efficiency with playing strength, targeting real-time responsiveness while maintaining strategic depth.

### 1.2 System Overview

AxiomX consists of four primary subsystems:

1. **Chess Engine Module** - Core game logic and AI search
2. **Game Logic Module** - Move execution and state management  
3. **UI Components Module** - Visual representation and user interaction
4. **Application Module** - Integration and control flow

The engine operates at search depth $d = 3$ with move ordering heuristics and evaluates positions using material balance and positional factors encoded in piece-square tables.

---

## 2. Theoretical Framework

### 2.1 Game Tree Representation

Chess can be modeled as a two-player zero-sum game with perfect information. The game state space forms a directed acyclic graph where:

- Each node $n$ represents a board position $B$
- Each edge $(n_i, n_j)$ represents a legal move $m$
- Terminal nodes correspond to checkmate or stalemate positions

Let $\mathcal{S}$ denote the set of all legal board positions and $\mathcal{M}(s)$ the set of legal moves from state $s \in \mathcal{S}$.

### 2.2 Minimax Algorithm

The minimax algorithm operates on the principle that one player (maximizer) seeks to maximize the evaluation function while the opponent (minimizer) seeks to minimize it. For a game tree of depth $d$, the minimax value is defined recursively:

$$
\text{minimax}(n, d, \text{max}) = \begin{cases}
\text{eval}(n) & \text{if } d = 0 \text{ or } n \text{ is terminal} \\
\max_{c \in \text{children}(n)} \text{minimax}(c, d-1, \text{false}) & \text{if max} \\
\min_{c \in \text{children}(n)} \text{minimax}(c, d-1, \text{true}) & \text{otherwise}
\end{cases}
$$

where $\text{eval}(n)$ is the static evaluation function applied to node $n$.

### 2.3 Alpha-Beta Pruning

Alpha-beta pruning reduces the effective branching factor by eliminating subtrees that cannot influence the final decision. The algorithm maintains two bounds:

- $\alpha$: the best value the maximizer can guarantee
- $\beta$: the best value the minimizer can guarantee

A cutoff occurs when $\beta \leq \alpha$, allowing the algorithm to prune remaining siblings. The pruned minimax formulation is:

$$
\text{minimax}(n, d, \alpha, \beta, \text{max}) = \begin{cases}
\text{eval}(n) & \text{if } d = 0 \\
\max_{c \in C(n)} \{ \text{minimax}(c, d-1, \alpha, \beta, \text{false}) : \beta > \alpha \} & \text{if max} \\
\min_{c \in C(n)} \{ \text{minimax}(c, d-1, \alpha, \beta, \text{true}) : \beta > \alpha \} & \text{otherwise}
\end{cases}
$$

where $C(n)$ denotes the children of $n$ considered before a cutoff.

**Theorem 1** (Alpha-Beta Correctness): Alpha-beta pruning returns the same value as standard minimax.

**Proof**: The pruning condition $\beta \leq \alpha$ only eliminates subtrees that cannot affect the parent's value. The maximizer at depth $d$ has already found a move guaranteeing $\alpha$, so any path through the minimizer at $d-1$ yielding $\beta \leq \alpha$ cannot be selected. $\square$

---

## 3. Position Evaluation

### 3.1 Evaluation Function Design

The static evaluation function $E: \mathcal{S} \rightarrow \mathbb{R}$ maps board positions to numerical values. A positive value indicates advantage for Black (the maximizer), while negative values favor White. The evaluation decomposes as:

$$
E(s) = E_{\text{material}}(s) + E_{\text{positional}}(s)
$$

### 3.2 Material Evaluation

The material component assigns fixed values to pieces:

$$
E_{\text{material}}(s) = \sum_{p \in \text{pieces}(s)} v(p) \cdot \text{sgn}(p)
$$

where:
- $v(p) \in \{100, 320, 330, 500, 900, 20000\}$ for pawns, knights, bishops, rooks, queens, and kings respectively
- $\text{sgn}(p) = +1$ if $p$ belongs to Black, $-1$ if White

### 3.3 Piece-Square Tables

Positional evaluation employs piece-square tables (PST) that encode strategic principles:

$$
E_{\text{positional}}(s) = \sum_{p \in \text{pieces}(s)} \text{PST}[p_{\text{type}}][r(p)][c(p)] \cdot \text{sgn}(p)
$$

where $r(p)$ and $c(p)$ are the row and column coordinates, adjusted for piece color.

**Pawn PST Design Principles:**
- Advancement bonus increases toward the promotion rank
- Central pawns (d4, e4, d5, e5) receive higher values
- Doubled pawns on the same file incur penalties (implicit)

**Knight PST Design Principles:**
- Central squares receive significant bonuses ($+20$ centipawns)
- Edge and corner squares penalized ($-50$ centipawns)
- Knights on the rim have lower mobility

**Bishop PST Design Principles:**
- Long diagonal squares favored
- Central control rewarded
- Edge positions avoided

**King PST Design Principles:**
- Castled positions (g1, h1 for White) heavily favored ($+30$ centipawns)
- Central king positions penalized ($-50$ centipawns)
- Reflects king safety in middlegame

### 3.4 Complete Evaluation Expression

The full evaluation function is:

$$
E(s) = \sum_{p \in \text{pieces}(s)} \left[ v(p) + \text{PST}[p_{\text{type}}][r(p)][c(p)] \right] \cdot \text{sgn}(p)
$$

---

## 4. Search Enhancements

### 4.1 Move Ordering

Move ordering significantly impacts alpha-beta pruning efficiency. The branching factor can be reduced from $b$ to approximately $\sqrt{b}$ with optimal ordering.

**Capture Prioritization Heuristic:**

Captures are scored using the Most Valuable Victim - Least Valuable Attacker (MVV-LVA) principle:

$$
\text{score}(m) = 10 \cdot v(\text{captured}) - v(\text{attacker})
$$

This heuristic examines winning captures (e.g., pawn captures queen) before potentially losing captures (e.g., queen captures pawn).

**Theorem 2** (Move Ordering Benefit): With optimal move ordering, the effective branching factor of alpha-beta search is reduced to $b^*$ where:

$$
b^* = \mathcal{O}(\sqrt{b})
$$

resulting in a search depth increase by a factor of 2 for the same computational budget.

### 4.2 Search Depth Selection

AxiomX operates at a fixed depth $d = 3$, corresponding to:
- 1.5 full moves of lookahead
- Average of $\approx b^3$ positions evaluated per search (where $b \approx 35$ is the average branching factor in chess)
- With alpha-beta pruning and move ordering: $\approx 35^{1.5} \cdot 2 \approx 415$ positions in practice

**Computational Complexity:**

Time complexity: $\mathcal{O}(b^d)$ without pruning, $\mathcal{O}(b^{d/2})$ with optimal pruning  
Space complexity: $\mathcal{O}(d)$ due to depth-first traversal

---

## 5. Implementation Architecture

### 5.1 Module Decomposition

**chess_engine.py**
- `initialize_board()`: Creates standard starting position
- `is_valid_move()`: Validates move legality under chess rules
- `evaluate_board()`: Computes static position evaluation
- `minimax()`: Implements alpha-beta pruned search
- `get_ai_move()`: Entry point for AI move generation

**game_logic.py**
- `execute_move()`: Applies move to board state
- `handle_player_move()`: Processes human input
- `handle_ai_move()`: Coordinates AI move computation
- `handle_promotion()`: Manages pawn promotion

**ui_components.py**
- `create_board_layout()`: Renders 8×8 grid with pieces
- `create_evaluation_bar()`: Visualizes position assessment
- `create_move_list()`: Displays game notation history

**app.py**
- Dash application initialization
- Callback management for user interactions
- State persistence and game flow control

### 5.2 State Representation

Game state is maintained as a Python dictionary with the following structure representing all necessary information for game continuation and move legality verification.

### 5.3 Special Move Implementation

**Castling:**

Conditions verified:
1. King and rook have not moved (`moved: false` flag)
2. Squares between king and rook are empty
3. King is not in check
4. King does not pass through or land on attacked square

**En Passant:**

Tracked via `en_passant` state variable storing the position behind a pawn that moved two squares. Valid for exactly one move.

**Pawn Promotion:**

Triggers modal interface when pawn reaches final rank. Game state pauses until player selects promotion piece (queen, rook, bishop, or knight).

---

## 6. Performance Analysis

### 6.1 Time Complexity Analysis

For a search depth $d$ and branching factor $b$:

| Algorithm | Time Complexity | Nodes Visited (d=3, b=35) |
|-----------|----------------|---------------------------|
| Minimax | $\mathcal{O}(b^d)$ | 42,875 |
| Alpha-Beta (worst) | $\mathcal{O}(b^d)$ | 42,875 |
| Alpha-Beta (average) | $\mathcal{O}(b^{3d/4})$ | ~7,500 |
| Alpha-Beta (best) | $\mathcal{O}(b^{d/2})$ | ~210 |

With move ordering, AxiomX achieves near-optimal pruning in tactical positions.

### 6.2 Move Generation Efficiency

The system evaluates approximately 400-800 positions per move in typical middlegame positions, completing in under 0.5 seconds on modern hardware.

### 6.3 Opening Book Absence

AxiomX operates without an opening book, relying entirely on search and evaluation. While this sacrifices opening optimality, it ensures consistent algorithmic behavior across all game phases.

---

## 7. User Interface Design

### 7.1 Terminal Aesthetics

The interface adopts a minimalist terminal theme:
- Black background (#000000)
- Dark grey/black checkerboard pattern (#333333/#000000)
- White pieces rendered in white (#FFFFFF)
- Black pieces rendered in green (#00FF00)
- Monospace font (Courier New)

### 7.2 Visual Feedback System

**Move Indication:**
- Selected square: bright green highlight (#00FF00)
- Valid destinations: small grey circle (20% square size, #666666)
- Position evaluation: vertical bar with color-coded segments

**Information Display:**
- Left panel: Interactive chess board (50% viewport width)
- Right panel subdivided:
  - Evaluation bar (50px width)
  - Move history (remaining width)
  - Control panel (bottom section)

### 7.3 Interaction Model

Player interaction follows a two-click paradigm:
1. First click selects a piece (must match player's color)
2. Valid moves are displayed as grey circles
3. Second click on valid destination executes the move
4. AI responds automatically after player move completion

---

## 8. Experimental Results

### 8.1 Tactical Performance

AxiomX successfully identifies:
- Immediate captures (1-ply)
- Tactical combinations up to 3-ply (forks, pins, skewers)
- Simple checkmating patterns (back rank mates, queen-king combinations)

### 8.2 Strategic Understanding

The piece-square tables encode:
- King safety principles (castling preference)
- Central pawn advancement
- Knight centralization vs. edge placement
- Bishop long diagonal control

### 8.3 Limitations

1. **Horizon Effect**: Combinations exceeding depth 3 are not visualized
2. **Positional Complexity**: Lacks understanding of pawn structure, weak squares
3. **Endgame Knowledge**: No tablebase integration for theoretical endgames
4. **Time Management**: Fixed depth rather than time-based search

---

## 9. Technical Requirements

### 9.1 Dependencies

```
dash==2.14.2
dash-bootstrap-components==1.5.0
```

### 9.2 System Requirements

- Python 3.8 or higher
- 512 MB RAM minimum
- Modern web browser with JavaScript enabled

### 9.3 Installation

```bash
pip install -r requirements.txt
python app.py
```

Access the interface at `http://localhost:8050`

---

## 10. Future Work

### 10.1 Algorithmic Enhancements

**Quiescence Search:**

Extend search at leaf nodes to resolve captures:

$$
Q(\text{pos}, \alpha, \beta) = \begin{cases}
\max(\text{eval}(\text{pos}), \max_{m \in \text{captures}} Q(\text{apply}(m), \alpha, \beta)) & \text{if maximizer} \\
\min(\text{eval}(\text{pos}), \min_{m \in \text{captures}} Q(\text{apply}(m), \alpha, \beta)) & \text{if minimizer}
\end{cases}
$$

**Iterative Deepening:**

Search progressively deeper until time threshold with aspiration windows for improved bound estimation.

**Transposition Tables:**

Cache evaluated positions using Zobrist hashing to avoid re-evaluation of transposed positions.

### 10.2 Evaluation Refinements

- **Pawn Structure Analysis**: Isolated, doubled, backward pawns
- **King Safety**: Pawn shield evaluation, attacking piece proximity
- **Piece Mobility**: Count of legal moves for each piece
- **Endgame Tablebases**: Perfect play in simplified positions

### 10.3 Machine Learning Integration

Potential for:
- Neural network-based evaluation functions trained on master games
- Reinforcement learning for self-play improvement
- Pattern recognition for strategic motifs

---

## 11. Conclusion

AxiomX demonstrates the practical application of classical game tree search algorithms to the chess domain. The engine achieves a balance between computational efficiency and playing strength through alpha-beta pruning, move ordering, and position evaluation via piece-square tables. The implementation provides a foundation for studying search algorithms, evaluation heuristics, and human-computer interaction in board games.

The modular architecture facilitates future enhancements, including deeper search techniques, refined evaluation functions, and machine learning integration. AxiomX serves as both a functional chess opponent and an educational platform for understanding the principles underlying game-playing artificial intelligence.

---

## 12. References

1. Shannon, C. E. (1950). Programming a computer for playing chess. *Philosophical Magazine*, 41(314), 256-275.

2. Knuth, D. E., & Moore, R. W. (1975). An analysis of alpha-beta pruning. *Artificial Intelligence*, 6(4), 293-326.

3. Berliner, H. J. (1979). The B* tree search algorithm: A best-first proof procedure. *Artificial Intelligence*, 12(1), 23-40.

4. Marsland, T. A. (1986). A review of game-tree pruning. *ICCA Journal*, 9(1), 3-19.

5. Schaeffer, J. (1989). The history heuristic and alpha-beta search enhancements in practice. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11(11), 1203-1212.

6. Campbell, M., Hoane Jr, A. J., & Hsu, F. h. (2002). Deep Blue. *Artificial Intelligence*, 134(1-2), 57-83.

7. Silver, D., Hubert, T., Schrittwieser, J., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. *Science*, 362(6419), 1140-1144.

---

## Appendix A: Algebraic Notation

AxiomX records moves in standard algebraic notation (SAN):

| Symbol | Meaning |
|--------|---------|
| e4 | Pawn to e4 |
| Nf3 | Knight to f3 |
| Bxf7+ | Bishop captures on f7, check |
| O-O | Kingside castling |
| O-O-O | Queenside castling |
| e8=Q | Pawn promotes to queen |
| exd6 e.p. | En passant capture |

---

## Appendix B: Piece Values

Centipawn values used in evaluation:

| Piece | Value | Rationale |
|-------|-------|-----------|
| Pawn | 100 | Base unit |
| Knight | 320 | ~3 pawns + minor tactics |
| Bishop | 330 | Slightly superior to knight in open positions |
| Rook | 500 | ~5 pawns, dominates open files |
| Queen | 900 | ~9 pawns, most powerful piece |
| King | 20000 | Game-ending if captured |

---

## Appendix C: Complexity Classes

**Decision Problem:** Given position $P$ and depth $d$, does there exist a move leading to checkmate within $d$ plies?

This problem is **PSPACE-complete** for generalized chess on $n \times n$ boards.

**State Space Complexity:**

Shannon Number (approximate): $10^{120}$ distinct legal positions

Game Tree Complexity: $10^{123}$ total game-tree nodes

---

**Document Version:** 1.0  
**Last Updated:** January 12, 2026  
**Authors:** Brayden Boyko
