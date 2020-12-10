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

[rubik-cube](https://github.com/pglass/cube) project will need to be installed in order to properly use the `greedy.py` project this can visited at the link or installed using pip via this command: pip install rubik-cube

To use the `greedy.py` file all you need to do is simply travel to the directory it is located in via a CLI and execute the following command: `py greedy.py` and it will start execution and solving cubes for as long as you desire. In order to stop the execution simply use CTRL+C in the command prompt or terminal and it will terminate.
