# Rubiks-Cube-Solver
Solving a Rubik's cube utilizing genetic and greedy algorithms

## How to start
Install python3 and use pip3 to install pycuber and matplotlib. Also 2 folders must be created, a `pictures` and a `data` folder.

### For GAs
To try and randomly guess a move set for a scramble cube use `python3 Gods\ number\ test.py numGens popSize numberOfMoves`.
This will utilize 1 thread and attempt to find a moveset given a number of generations, population size and number of moves.

To use GAs to solve a cube step by step, like [here](https://rubiks-cube-solver.com/how-to-solve/) use `python3 GA\ Multi\ Core.py numGens popSize numberOfMoves numThreads`, this file only works on Linux. This takes the same parameters as above however it adds on the number of threads you wish to use. This helps in speeding up the program as the fitness function is more complicated than `Gods number test.py` and allows for bigger move sets or bigger population sizes.

`GA Output` contains a sample output of files and text for running `GA Multi Core.py` utilizing this command `python3 GA\ Multi\ Core.py 100 100 100 16`. For commented code as well, `GA Multi Core.py` is considered the main file for GAs and is fully commented.

### For greedy

