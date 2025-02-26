import math

class TicTacToe:
    def __init__(self):
        # Initialize a 3x3 board with '_' and set the first player to 'X'
        self.board = [['_' for _ in range(3)] for _ in range(3)]

    def choose(self, row, col, player):
        # Place 'X' or 'O' if the spot is available
        if self.board[row][col] == '_':
            self.board[row][col] = player
            return True
        return False

    def get_empty_cells(self):
        # Return a list of all available cells as (row, col) tuples
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == '_']

    def did_win(self, player):
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if all(self.board[i][j] == player for j in range(3)) or \
               all(self.board[j][i] == player for j in range(3)):
                return True
        if all(self.board[i][i] == player for i in range(3)) or \
           all(self.board[i][2 - i] == player for i in range(3)):
            return True
        return False

    def did_tie(self):
        # Check for a tie (no empty spaces and no winner)
        return all(cell != '_' for row in self.board for cell in row) and \
               not (self.did_win('X') or self.did_win('O'))

    def minimax(self, is_maximizing):
        # Minimax algorithm to calculate the best move for 'O'
        if self.did_win('X'):
            return -1  # X wins
        elif self.did_win('O'):
            return 1  # O wins
        elif self.did_tie():
            return 0  # Tie

        if is_maximizing:
            best_score = -math.inf
            for row, col in self.get_empty_cells():
                self.board[row][col] = 'O'
                score = self.minimax(False)
                self.board[row][col] = '_'
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = math.inf
            for row, col in self.get_empty_cells():
                self.board[row][col] = 'X'
                score = self.minimax(True)
                self.board[row][col] = '_'
                best_score = min(score, best_score)
            return best_score

    def best_move(self):
        # Find the best move for 'O' (Minimax)
        best_score = -math.inf
        move = None
        for row, col in self.get_empty_cells():
            self.board[row][col] = 'O'
            score = self.minimax(False)
            self.board[row][col] = '_'
            if score > best_score:
                best_score = score
                move = (row, col)
        return move

    def __str__(self):
        # Return the board as a string for display
        return '\n'.join([' '.join(row) for row in self.board])

class TicTacToeMain:
    def main(self):
        game = TicTacToe()
        print("Play a game of Tic Tac Toe")
        
        while True:
            print(game)
            
            # Player X (Human) makes a move
            if not game.get_empty_cells() or game.did_win('O'):
                break  # Stop if no moves are left or O wins

            while True:
                try:
                    print("Row and column for X? ", end="")
                    row, col = map(int, input().split())
                    if game.choose(row, col, 'X'):
                        break  # Valid move made
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Enter two integers between 0 and 2.")

            # Check if X won or if the game ended in a tie
            if game.did_win('X') or game.did_tie():
                break

            # Computer (O) makes the best move using Minimax
            print("Computer (O) is making a move...")
            row, col = game.best_move()
            game.choose(row, col, 'O')

            # Check if O won or if the game ended in a tie
            if game.did_win('O') or game.did_tie():
                break

        print(game)
        if game.did_win('X'):
            print("Game Over. X won!")
        elif game.did_win('O'):
            print("Game Over. O won!")
        else:
            print("Game Over. It's a tie!")

if __name__ == "__main__":
    TicTacToeMain().main()
