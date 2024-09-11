import argparse
import os
import ray
import supersuit as ss
from PIL import Image
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
from pettingzoo.butterfly import pistonball_v6
from datetime import datetime

# Define a custom CNN-based model inheriting from TorchModelV2
class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        # Initialize parent classes
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        # Define CNN layers for feature extraction
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),  # First convolutional layer
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),  # Second convolutional layer
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),  # Third convolutional layer
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output for fully connected layers
            nn.Linear(3136, 512),  # Fully connected layer
            nn.ReLU(),
        )
        # Output layers for policy and value function
        self.policy_fn = nn.Linear(512, num_outputs)  # Policy layer
        self.value_fn = nn.Linear(512, 1)  # Value function layer

    # Forward pass through the network
    def forward(self, input_dict, state, seq_lens):
        # Rearrange the observation tensor and pass through CNN layers
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)  # Compute value function
        return self.policy_fn(model_out), state  # Return policy logits and state

    # Return the value function output
    def value_function(self):
        return self._value_out.flatten()


# Set SDL environment variable to avoid errors when rendering without a display
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Argument parser to allow the user to specify the checkpoint path
parser = argparse.ArgumentParser(
    description="Render pretrained policy loaded from checkpoint"
)
parser.add_argument(
    "--checkpoint-path",
    help="Path to the checkpoint. This path will likely be something like this: `~/pistonball_results/pistonball_v6/PPO/PPO_pistonball_v6_660ce_00000_0_2021-06-11_12-30-57/checkpoint_000050/checkpoint-50`",
)

# Parse command-line arguments
args = parser.parse_args()

# Exit if the checkpoint path is not provided
if args.checkpoint_path is None:
    print("The following arguments are required: --checkpoint-path")
    exit(0)

# Expand the user's checkpoint path
checkpoint_path = os.path.expanduser(args.checkpoint_path)

# Register the custom CNN model with RLlib
ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)


# Environment creator function for pistonball
def env_creator():
    # Create the pistonball environment with specified parameters
    env = pistonball_v6.env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
        render_mode="rgb_array",  # Enable rendering in RGB array mode for visualization
    )
    # Preprocess the environment using SuperSuit wrappers
    env = ss.color_reduction_v0(env, mode="B")  # Convert to black-and-white
    env = ss.dtype_v0(env, "float32")  # Convert data type to float32
    env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize frames to 84x84
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)  # Normalize observations
    env = ss.frame_stack_v1(env, 3)  # Stack frames for temporal context
    return env


# Create the environment and register it
env = env_creator()
env_name = "pistonball_v6"
register_env(env_name, lambda config: PettingZooEnv(env_creator()))  # Register with RLlib

# Initialize Ray
ray.init()

# Load the trained PPO agent from the checkpoint
PPOagent = PPO.from_checkpoint(checkpoint_path)

# Initialize variables for storing rewards and rendered frames
reward_sum = 0
frame_list = []
i = 0
env.reset()  # Reset the environment before starting

# Run the agent in the environment, collecting observations and rewards
for agent in env.agent_iter():
    # Retrieve the latest observation, reward, termination, truncation, and info
    observation, reward, termination, truncation, info = env.last()
    reward_sum += reward  # Accumulate rewards
    if termination or truncation:  # If the episode has terminated or truncated
        action = None
    else:
        action = PPOagent.compute_single_action(observation)  # Compute the action for the agent

    env.step(action)  # Take the action in the environment
    i += 1
    # Render the environment and save the frame after each full cycle through agents
    if i % (len(env.possible_agents) + 1) == 0:
        img = Image.fromarray(env.render())  # Convert rendered array to image
        frame_list.append(img)  # Append the frame to the list
env.close()  # Close the environment after execution

# Ensure the results directory exists, create if it doesn't
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Generate a timestamp to use for the GIF filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
gif_filename = os.path.join(results_dir, f"pistonball_{timestamp}.gif")

# Save the frames as a GIF in the results directory with the timestamped name
frame_list[0].save(
    gif_filename, save_all=True, append_images=frame_list[1:], duration=3, loop=0
)

# Output the final accumulated reward and the path of the saved GIF
print(f"reward_sum: {reward_sum}")
print(f"GIF saved as {gif_filename}")