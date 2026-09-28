import numpy as np
import cv2 as cv
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

#Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Constants
GRID_SIZE = 50
CELL_SIZE = 10
IMG_SIZE = GRID_SIZE * CELL_SIZE
INPUT_DIM = GRID_SIZE * GRID_SIZE * 3
HIDDEN_DIM = 512
ACTION_DIM = 4

#Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 8000
TARGET_UPDATE = 500
MEMORY_SIZE = 10000
LEARNING_RATE = 0.0005


#1.Neural Network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, 256)
        self.out = nn.Linear(256, ACTION_DIM)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


#2.Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


#3.Environment
class GridEnvironment:
    def __init__(self):
        self.img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        self.grid_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.reset()

    def reset(self):
        self.img.fill(0)
        self.grid_map = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.99, 0.01])

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.grid_map[r, c] == 1:
                    self.draw_cell(r, c, (255, 0, 0))  # obstacle

        empty_spots = list(zip(*np.where(self.grid_map == 0)))
        if len(empty_spots) < 2:
            return self.reset()

        idx_s, idx_f = random.sample(empty_spots, 2)
        self.agent_pos = list(idx_s)
        self.goal_pos = list(idx_f)

        self.steps = 0
        self.update_render()
        return self.get_flattened_state()

    def draw_cell(self, r, c, color):
        y, x = r * CELL_SIZE, c * CELL_SIZE
        self.img[y:y + CELL_SIZE, x:x + CELL_SIZE] = color

    def update_render(self):
        self.draw_cell(self.goal_pos[0], self.goal_pos[1], (0, 0, 255))  # goal
        self.draw_cell(self.agent_pos[0], self.agent_pos[1], (0, 255, 0))  # agent

    def clear_agent(self):
        self.draw_cell(self.agent_pos[0], self.agent_pos[1], (0, 0, 0))

    def get_flattened_state(self):
        state = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)
        state[:, :, 0] = self.grid_map
        state[self.agent_pos[0], self.agent_pos[1], 1] = 1.0
        state[self.goal_pos[0], self.goal_pos[1], 2] = 1.0
        return state.flatten()

    def step(self, action_idx):
        self.steps += 1
        self.clear_agent()

        # OLD distance
        old_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + \
                   abs(self.agent_pos[1] - self.goal_pos[1])

        dr, dc = self.actions[action_idx]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        done = False

        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            if self.grid_map[nr, nc] == 1:
                reward = -5  # obstacle
            else:
                self.agent_pos = [nr, nc]

                # NEW distance
                new_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + \
                           abs(self.agent_pos[1] - self.goal_pos[1])

                # 🔥 Direction-based reward
                if new_dist < old_dist:
                    reward = +1.0   # closer
                elif new_dist > old_dist:
                    reward = -1.0   # farther
                else:
                    reward = -0.2   # no change

                reward -= 0.05  # step penalty

                if self.agent_pos == self.goal_pos:
                    reward = 25
                    done = True
        else:
            reward = -5  # wall hit

        if self.steps > 1200:
            done = True

        self.update_render()
        return self.get_flattened_state(), reward, done


#4.Setup
env = GridEnvironment()

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer(MEMORY_SIZE)

steps_done = 0


def select_action(state):
    global steps_done

    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    np.exp(-1. * steps_done / EPS_DECAY)

    steps_done += 1

    if random.random() > eps_threshold:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            return policy_net(state_t).argmax(dim=1).item()
    else:
        return random.randrange(ACTION_DIM)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    state_batch = torch.FloatTensor(np.array(states)).to(device)
    action_batch = torch.LongTensor(actions).unsqueeze(1).to(device)
    reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_state_batch = torch.FloatTensor(np.array(next_states)).to(device)
    done_batch = torch.FloatTensor(dones).unsqueeze(1).to(device)

    current_q_values = policy_net(state_batch).gather(1, action_batch)

    with torch.no_grad():
        next_max_q = target_net(next_state_batch).max(1)[0].unsqueeze(1)
        expected_q_values = reward_batch + (GAMMA * next_max_q * (1 - done_batch))

    loss = F.mse_loss(current_q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#5.Training Loop
print("Training DQN...")

episode = 0

while True:
    state = env.reset()
    total_reward = 0

    while True:
        action = select_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward

        memory.push(state, action, reward, next_state, done)
        state = next_state

        optimize_model()

        cv.imshow('DQN Training', env.img)
        if cv.waitKey(1) == 27:
            cv.destroyAllWindows()
            exit()

        if done:
            break

    if steps_done % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Reward: {total_reward:.2f}")
    episode += 1
