import pygame
import numpy as np
import random
import matplotlib.pyplot as plt

# Define five different mazes with dimensions 7x7
mazes = [
    np.array([
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]),
    np.array([
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0]
    ]),
    np.array([
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0]
    ]),
    np.array([
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0]
    ])
]

# Constants
WIDTH, HEIGHT = 700, 700  # Adjusted for a 7x7 maze
CELL_SIZE = WIDTH // 7  # Assuming all mazes are 7x7

# Initialise Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Reinforcement Learning Maze")

def draw_maze(maze):
    rows, cols = maze.shape
    for row in range(rows):
        for col in range(cols):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = (255, 255, 255) if maze[row, col] == 0 else (0, 0, 0)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)

class MazeEnv:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.state = start
        self.actions = ['up', 'down', 'left', 'right']
        self.rows, self.cols = maze.shape

    def reset(self):
        self.state = self.start
        return self.state

    def set_maze(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.rows, self.cols = maze.shape
        self.state = self.start

    def step(self, action):
        row, col = self.state
        if action == 'up':
            row = max(row - 1, 0)
        elif action == 'down':
            row = min(row + 1, self.rows - 1)
        elif action == 'left':
            col = max(col - 1, 0)
        elif action == 'right':
            col = min(col + 1, self.cols - 1)

        if self.maze[row, col] == 1:
            next_state = self.state  # Wall hit, stay in place
        else:
            next_state = (row, col)

        reward = -1  # Penalty for each move
        done = False
        if next_state == self.goal:
            reward = 100  # Reward for reaching the goal
            done = True

        self.state = next_state
        return next_state, reward, done

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        return self.q_table.get(state, {}).get(action, 0.0)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = {action: self.get_q_value(state, action) for action in self.actions}
            max_q = max(q_values.values())
            actions_with_max_q = [action for action, q in q_values.items() if q == max_q]
            return random.choice(actions_with_max_q)

    def learn(self, state, action, reward, next_state):
        old_q = self.get_q_value(state, action)
        future_q = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)
        self.q_table.setdefault(state, {})[action] = new_q

def main():
    # Define the starting and goal positions for each maze
    start_positions = [(0, 0)] * 5  # Starting position for each maze
    goal_positions = [(6, 6)] * 5   # Goal position for each maze

    # Initialize environment with the first maze
    env = MazeEnv(mazes[0], start_positions[0], goal_positions[0])
    agent = QLearningAgent(actions=env.actions)

    episodes_per_maze = 500
    num_mazes = len(mazes)
    total_episodes = episodes_per_maze * num_mazes

    # Data collection lists
    total_rewards = []
    steps_per_episode = []
    epsilon_values = []

    # For plotting purposes, keep track of maze changes
    maze_changes = []

    for episode in range(total_episodes):
        # Switch to a new maze after every 500 episodes
        if episode % episodes_per_maze == 0 and episode != 0:
            maze_index = (episode // episodes_per_maze) % num_mazes
            env.set_maze(mazes[maze_index], start_positions[maze_index], goal_positions[maze_index])
            print(f"Switched to Maze {maze_index + 1}")
            maze_changes.append(episode)

        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            pygame.event.pump()

            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

            total_reward += reward
            steps += 1

            # Visualisation
            screen.fill((255, 255, 255))
            draw_maze(env.maze)

            # Draw the agent
            row, col = env.state
            agent_rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (0, 0, 255), agent_rect)

            # Draw the goal
            goal_row, goal_col = env.goal
            goal_rect = pygame.Rect(goal_col * CELL_SIZE, goal_row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (0, 255, 0), goal_rect)

            pygame.display.flip()
            pygame.time.delay(1)

        # Data collection
        total_rewards.append(total_reward)
        steps_per_episode.append(steps)
        epsilon_values.append(agent.epsilon)

        # Decay epsilon
        if agent.epsilon > 0.01:
            agent.epsilon *= 0.995

        # Optional: Print episode info
        print(f"Episode {episode+1}/{total_episodes} completed. Total Reward: {total_reward}, Steps: {steps}")

    pygame.quit()

    # Plotting the training progress with separate graphs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Upper Graph: Total Reward and Steps per Episode
    color1 = 'tab:blue'
    ax1.set_ylabel('Total Reward', color=color1)
    line1 = ax1.plot(total_rewards, label='Total Reward', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Create a second y-axis on the upper graph for Steps per Episode
    ax1b = ax1.twinx()
    color2 = 'tab:green'
    ax1b.set_ylabel('Steps per Episode', color=color2)
    line2 = ax1b.plot(steps_per_episode, label='Steps per Episode', color=color2)
    ax1b.tick_params(axis='y', labelcolor=color2)

    # Add maze change indicators to the upper graph
    for change in maze_changes:
        ax1.axvline(x=change, color='gray', linestyle='--', label='Maze Change' if change == maze_changes[0] else "")

    # Combine legends for the upper graph
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    # Lower Graph: Epsilon Value per Episode
    color3 = 'tab:red'
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon Value', color=color3)
    line3 = ax2.plot(epsilon_values, label='Epsilon Value', color=color3)
    ax2.tick_params(axis='y', labelcolor=color3)

    # Add maze change indicators to the lower graph
    for change in maze_changes:
        ax2.axvline(x=change, color='gray', linestyle='--', label='Maze Change' if change == maze_changes[0] else "")

    # Legend for the lower graph
    ax2.legend(loc='upper right')

    # Set titles
    ax1.set_title('Training Progress: Total Reward and Steps per Episode')
    ax2.set_title('Epsilon Value per Episode')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
