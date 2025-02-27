# Machine_Learning
A compilation of some of the Machine Learning projects and assignments that I have worked on

## Income assignment

This mini-project was given to explore how different models classify a dataset that was given (INCOME.CSV).

The 4 models used were: Naive-Bayes, k-NN, Decision Trees and Random Forest. But before using these models, it was crucial to clean the data. Data cleaning is a crucial step in Machine Learning where we get rid of unusable, missing or irrelevant data.

After modelling and training, it was found that k-NN and Naive-Bayes had the highest accuracy in classifying the data given😮. Theoretically, Random Forest should have had the highest accuracy but the other two aforementioned models performed better likely due to its strong assumptions on feature independence.

As a start, this was a really cool project in opening the doors to Machine Learning. However, future work should explore more advanced techinques in redefining hyperparameters better so as to get more acuurate sccores

## Tic-Tac-Toe

Takes me back to my childhood; playing tic-tac-toe as the teacher was scribbling on the board😥😂
I then got the chance to implement a program where this legend of a game could be played on a computer.

There are two parts to the code: tictactoe.py and main.py:
1. tictactoe.py- Minimax-Based
   This was an implementation of the minimax algorithm for an AI opponent that takes O. The game starts with a       human (X) playing against the AI. The AI evaluates all pssible future moves and selects the optimal move.
2. main.py- Two-Player Tic-Tac-Toe
   This is a "control" program that simulates the normal human vs human version of the game.

## Deep Neural Networks for Hand-Written Digit Recognition (kminst.py and emnist.py)

It's a mouthful, isn't it?😂 
This was a project aimed at implementing a deep neural network from scratch in python that can be trained to recognize hand-written digit images. The sigmoid activation function of the neurons were adapted to simulate the scaled hyperbolic tangent function.

To demostrate the program's accuracy, training was done to recognize at least 4 digits from the MNIST dataset with a 90% or more accuracy.

The second part (kminst.py) was an application of loop optimization to improve the execution time of the program. This included manipulation of array access patterns, loop blocking and loop unrolling.
