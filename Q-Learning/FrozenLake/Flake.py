import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict


# Create env
env = gym.make("FrozenLake-v1",
                desc = ["SFFF", "FHFH", "FFFH", "HFFG"],
                is_slippery = False,
                #render_mode = 'human',
)

# Actions
action_names = {0:'Left', 1:'Down', 2:'Right', 3:'Up'}
action_space_size = env.action_space.n

# States
state_space_size = env.observation_space.n

print('Environment:\n{}\n States: {}\n Actions:{}'.format(env.get_wrapper_attr('desc'),state_space_size, action_space_size))

class FLakeAgent:
    def __init__(self,
                 env: gym.Env,
                 learning_rate: float,
                 initial_epsilon: float,
                 epsilon_decay: float,
                 final_epsilon: float,
                 discount_factor:  float=0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor #importance of future rewards

        # Exploration params
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []
    
    def get_action(self, obs: tuple[int,int,bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0:'Left', 1:'Down', 2:'Right', 3:'Up'
        """
        # With prob=epsilon: explore
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # With prob (1-epsilon): exploit
        else:
            return int(np.argmax(self.q_values[obs]))
    
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int,int,bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



# Training hyperparameters:
learning_rate = 0.01                                   # Higher = faster but less stable
n_episodes = 1000                                  # Number of hands to practice
start_epsilon = 1.0                                    # Start with random actions
epsilon_decay = start_epsilon / (n_episodes / 2)       # Reduce exploration over time
final_epsilon = 0.1                                    # Always keep some exploration

# Create agent and env

env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = FLakeAgent(
                 env=env,
                 learning_rate=learning_rate,
                 initial_epsilon=start_epsilon,
                 epsilon_decay= epsilon_decay,
                 final_epsilon=final_epsilon,
)

# Training loop
from tqdm import tqdm # Progress bar

for episode in tqdm(range(n_episodes)):
    # New hand
    obs, info = env.reset()
    done = False

    # Play one hand
    while not done:
        # Agent choses action
        action = agent.get_action(obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Learn from this experience
        agent.update(obs,action,reward,terminated,next_obs)

        # Move to next state
        done = terminated or truncated
        obs = next_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()


from seaborn import heatmap
import pandas as pd

q_table = agent.q_values.values()

data = pd.DataFrame(data=q_table, columns=['Left', 'Down', 'Right', 'Up'])
heatmap(data, cmap='Blues', annot=True, square=False)
plt.ylabel('States')
plt.show()
