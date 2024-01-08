# CS 461 - Artificial Intelligence Term Project - Group 12

## 2048 w/ Min-Max Tree Search, Monte Carlo Tree Search, and Double Deep Q-Network (DDQN)

_Ceren Akyar, Deniz Mert Dilaverler, Berk Çakar, Elifsena Öz, İpek Öztaş_

---

## Running the algorithms:

1. First, ensure that you are running on Python 3.7. The algorithms are guaranteed to work on Python 3.7.
2. Then, activate your virtual environment using conda, venv, or any other virtual environment manager.
3. Install the requirements by running: `pip install -r requirements.txt`.
4. Run the algorithms using the following commands:
   - For Min-Max Tree Search: `python 2048_minmax.py`
   - For Monte Carlo Tree Search: `python 2048_mcts.py`
   - For DDQN: `python 2048_ddqn.py`

   - You can adjust the parameters by passing them as arguments. For example, you can run the Min-Max Tree Search algorithm with a depth of 3 via: `python 2048_minmax.py --depth 3`.
   - To see the full list of parameters, you can run: `python 2048_minmax.py --help`, `python 2048_mcts.py --help`, or `python 2048_ddqn.py --help`.

---

## Contributions by each team member:

- **Ceren Akyar:** MCTS experiments, plotting & inferring the results
- **Deniz Mert Dilaverler:** Implementation of Min-Max Tree Search and its experiments
- **Berk Çakar:** Implementation of MCTS, DDQN algorithms
- **Elifsena Öz:** DDQN experiments, plotting & inferring the results
- **İpek Öztaş:** DDQN experiments, plotting & inferring the results

---

## Overlapping parts with other course projects or research

**Note:** For this course, we were exposed to new concepts and algorithms that we have never covered in any of our courses before. Therefore, this project does not have any overlapping parts with any of our previous projects. In that sense, the algorithms we learned in CS 461 were re-implemented. Additionally, our knowledge from the CS 461 lectures helped us analyze the algorithms and their behaviors and interpret the plots.
