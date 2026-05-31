import gymnasium as gym
from gymnasium import spaces


class BairdCounterExample(gym.Env):
    """
    Baird's Counterexample

    This environment has:
    - 7 discrete states from 0 to 7.
    - 2 discrete states 0=dashed and 1=solid.
    - reward=0.
    - The initial state follows a uniform distribution.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    def __init__(self, render_mode=None):
        super().__init__()
        self.num_states = 7
        self.num_actions = 2
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)
        self.render_mode = render_mode
        self._state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state = self.np_random.integers(0, self.num_states)
        observation = self._state
        info = {}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if action == 0:
            new_state = self.np_random.integers(0, self.num_states - 1)
        else:
            new_state = self.num_states - 1
        self._state = new_state
        reward = 0.0
        terminated = False
        truncated = False # gym.make 会自动处理这个
        info = {}
        if self.render_mode == "human":
            self.render()
        return self._state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            print(f"Current state: {self._state}")

    def close(self):
        pass


# --- Numerical example ---

# if __name__ == "__main__":

#     env = BairdCounterExample(render_mode="human")
#     observation, info = env.reset(seed=42)
#     print(f"Init state: {observation}")
#     print("\n===== Running 5 steps =====")
#     for step_count in range(5):
#         action = env.action_space.sample()
#         observation, reward, terminated, truncated, info = env.step(action)
#         print(f"--- step: {step_count + 1} ---")
#         print(f"action: {action}")
#         print(f"new state: {observation}")
#         print(f"reward: {reward}")
#         if terminated or truncated:
#             print("Episode terminated, reset environment.")
#             observation, info = env.reset()
#     env.close()
#     print("\n===== Simulation ends, environment closed =====")
