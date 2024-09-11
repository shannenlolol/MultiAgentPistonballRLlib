# Multi-Agent Pistonball with Ray RLlib

This project implements multi-agent reinforcement learning using Ray's RLlib and the PettingZoo `pistonball` environment. The project uses the PPO (Proximal Policy Optimization) algorithm to train agents to cooperate in the Pistonball game. The game features multiple agents controlling pistons to move a ball towards a goal.

<br>

## Project Structure

- `multi_agent_pistonball_train.py`: The main script that trains the PPO agent on the Pistonball environment and saves the model and results.
- `multi_agent_pistonball_run.py`: A script that runs the trained agent and saves a GIF of the gameplay.
- `requirements.txt`: Contains the list of dependencies required for this project.
- `venv/`: (Optional) A virtual environment to run the project in an isolated environment.
- `pistonball_results/`: Directory where training results and checkpoints are saved (created during execution).
- `results/`: Directory where gameplay GIFs are saved (created during execution).

<br>

## Setup Instructions

### 1. Clone the repository

```
git clone 
cd multi_agent_pistonball
```

<br>

### 2. Install Dependencies

You will need to install the required Python libraries. First, ensure you have `pip` installed and then run the following command inside your virtual environment:

```bash
pip install -r requirements.txt
```

This will install Ray, PettingZoo, SuperSuit, and other dependencies needed for the project.

<br>

### 3. Activate Virtual Environment

If you haven't already, activate the virtual environment where the dependencies are installed:

On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

<br>

### 4. Train the Agent

To train the PPO agent using Ray's RLlib, run the following command:

```bash
python multi_agent_pistonball_train.py
```

This will start training the agent and save the results (including checkpoints) to the `pistonball_results/` directory. You can monitor the training progress in the console output.

<br>

### 5. Run the Trained Agent

Once the agent has been trained, you can visualize the gameplay by running the agent and saving a GIF of the agent's performance. Run the following command:

```bash
python multi_agent_pistonball_run.py --checkpoint-path <path_to_checkpoint>
```

Replace `<path_to_checkpoint>` with the path to a checkpoint saved in the `pistonball_results/` directory. The script will generate a GIF of the gameplay and save it to the `results/` directory.

<br>

### 6. View the Results

Check the `results/` directory for the generated GIF, which shows the trained agents playing the Pistonball game.

<br>

## Configuration

The training script allows customization of various hyperparameters such as learning rate, batch size, and number of workers. These parameters are defined in the `PPOConfig()` object inside `multi_agent_pistonball_train.py`. You can modify these values to experiment with different training setups.
