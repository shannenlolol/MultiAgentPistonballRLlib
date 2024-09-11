import os
import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
from pettingzoo.butterfly import pistonball_v6


# Define a custom CNN model for RLlib that processes image-based observations
class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        # Initialize the TorchModelV2 and nn.Module base classes
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)

        # Define the CNN layers
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),  # First convolution layer
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),  # Second convolution layer
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),  # Third convolution layer
            nn.ReLU(),
            nn.Flatten(),  # Flatten the output to feed into fully connected layers
            nn.Linear(3136, 512),  # Fully connected layer with 512 units
            nn.ReLU(),
        )
        # Output layers for policy and value function
        self.policy_fn = nn.Linear(512, num_outputs)  # Policy output layer
        self.value_fn = nn.Linear(512, 1)  # Value function output layer

    # Forward pass through the network
    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))  # Reorder the input for CNN
        self._value_out = self.value_fn(model_out)  # Compute value function output
        return self.policy_fn(model_out), state  # Return the policy logits and state

    # Value function for PPO's critic network
    def value_function(self):
        return self._value_out.flatten()


# Function to create the Pistonball environment with custom preprocessing
def env_creator(args):
    # Create the environment
    env = pistonball_v6.parallel_env(
        n_pistons=20,  # Number of pistons in the environment
        time_penalty=-0.1,  # Time penalty for each step
        continuous=True,  # Use continuous action space
        random_drop=True,  # Randomly drop pistons
        random_rotate=True,  # Randomly rotate pistons
        ball_mass=0.75,  # Set the ball's mass
        ball_friction=0.3,  # Set the ball's friction
        ball_elasticity=1.5,  # Set the ball's elasticity
        max_cycles=125,  # Maximum number of steps in an episode
    )
    # Preprocess the environment using SuperSuit wrappers
    env = ss.color_reduction_v0(env, mode="B")  # Convert to grayscale
    env = ss.dtype_v0(env, "float32")  # Convert observation data type to float32
    env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize observations to 84x84
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)  # Normalize observations to [0,1]
    env = ss.frame_stack_v1(env, 3)  # Stack frames for temporal information
    return env


if __name__ == "__main__":
    # Initialize Ray
    ray.init()

    # Environment name
    env_name = "pistonball_v6"

    # Register the environment for use with RLlib
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    
    # Register the custom CNN model with RLlib
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    # Configure the PPO algorithm
    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)  # Specify the environment and clip actions
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)  # Set up rollout workers and fragment length
        .training(
            train_batch_size=512,  # Training batch size
            lr=2e-5,  # Learning rate
            gamma=0.99,  # Discount factor for future rewards
            lambda_=0.9,  # Lambda for GAE (Generalized Advantage Estimation)
            use_gae=True,  # Use GAE for advantage estimation
            clip_param=0.4,  # PPO clip parameter
            grad_clip=None,  # No gradient clipping
            entropy_coeff=0.1,  # Coefficient for entropy bonus
            vf_loss_coeff=0.25,  # Coefficient for value function loss
            sgd_minibatch_size=64,  # Minibatch size for SGD updates
            num_sgd_iter=10,  # Number of SGD iterations per training batch
        )
        .debugging(log_level="ERROR")  # Set log level
        .framework(framework="torch")  # Specify PyTorch as the deep learning framework
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))  # Use GPUs if available
    )

    # Run the PPO algorithm with the specified configuration
    tune.run(
        "PPO",  # Use the PPO algorithm
        name="PPO",  # Name of the run
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},  # Total training timesteps
        checkpoint_freq=5,  # Frequency of saving checkpoints
        storage_path="/Users/shannenlee/Documents/multi_agent_pistonball/pistonball_results/" + env_name,  # Path to store results
        config=config.to_dict(),  # Convert the config to a dictionary and pass it
    )