CS 461 - Artificial Intelligence Term Project - Group 12
2048 w/ Min-Max Tree Search, Monte Carlo Tree Search, and Double Deep Q-Network (DDQN)
Ceren Akyar, Deniz Mert Dilaverler, Berk Çakar, Elifsena Öz, İpek Öztaş
================================================================================

Running the algorithms:

- First, you need to ensure that you are running on Python 3.7. The algorithms are guaranteed to work on Python 3.7.
- Then, activate your virtual environment via using conda, venv, or any other virtual environment manager.
- Install the requirements via using `pip install -r requirements.txt`.
- Run the algorithms via:
    For Min-Max Tree Search: `python 2048_minmax.py`
    For Monte Carlo Tree Search: `python 2048_mcts.py`
    For DDQN: `python 2048_ddqn.py`
        - You can adjust the parameters by passing them as arguments. For example, you can run the Min-Max Tree Search algorithm with a depth of 3 via `python 2048_minmax.py --depth 3`.
        - To see the full list of parameters, you can run `python 2048_minmax.py --help`, `python 2048_mcts.py --help`, or `python 2048_ddqn.py --help`.
