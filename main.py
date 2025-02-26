class TicTacToe:
    def __init__(self):
        # Initialize a 3x3 board with '_' and set the first player to 'X'
        self.board = [['_' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'

    def choose(self, row, column):
        # Validate the move
        if 0 <= row < 3 and 0 <= column < 3 and self.board[row][column] == '_':
            self.board[row][column] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

    def getNextPlayerChar(self):
        # Return the next player ('X' or 'O')
        return self.current_player

    def getCharArray(self):
        # Return the board state as a 2D array
        return self.board

    def __str__(self):
        # Return the board as a string for display
        return '\n'.join([' '.join(row) for row in self.board])

    def didWin(self, playerChar):
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if all(self.board[i][j] == playerChar for j in range(3)) or \
               all(self.board[j][i] == playerChar for j in range(3)):
                return True
        if all(self.board[i][i] == playerChar for i in range(3)) or \
           all(self.board[i][2 - i] == playerChar for i in range(3)):
            return True
        return False

    def didTie(self):
        # Check for a tie: no empty spaces and no winner
        return all(cell != '_' for row in self.board for cell in row) and \
               not (self.didWin('X') or self.didWin('O'))

    def notDone(self):
        # Check if the game is still ongoing
        return not self.didWin('X') and not self.didWin('O') and \
               any(cell == '_' for row in self.board for cell in row)

class TicTacToeMain:
    def main(self):
        print("Play a game of Tic Tac Toe")
        game = TicTacToe()

        # Play until there's a win or tie
        while game.notDone():
            print(game)
            print(f"Row and column for {game.getNextPlayerChar()}? ", end="")

            # Loop until valid input is received
            while True:
                user_input = input().strip()
                try:
                    row, col = map(int, user_input.split())
                    if game.choose(row, col):
                        break  # Valid move made, exit input loop
                    else:
                        print("Invalid move. Spot already taken or out of range. Try again.")
                except ValueError:
                    print("Invalid input. Please enter two integers separated by a space (e.g., '1 1').")
                except IndexError:
                    print("Row and column must be between 0 and 2. Try again.")

        # Display the final state and result
        print(game)
        if game.didWin('X'):
            print("Game Over\nX won")
        elif game.didWin('O'):
            print("Game Over\nO won")
        else:
            print("Game Over\nIt's a tie")

if __name__ == "__main__":
    TicTacToeMain().main()
