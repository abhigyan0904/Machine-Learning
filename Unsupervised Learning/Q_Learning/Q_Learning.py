import numpy as np
import cv2 as cv
import random

# Set up environment
img = np.zeros((500, 500, 3), dtype = np.uint8)
step_dict = {}
i = 0

for r in range(0, 500, 10):
    for c in range(0, 500, 10):
        step_dict[i] = img[r:r+10, c:c+10]
        i += 1

walls = set()
for i in range(0, 2500, 50):
    walls.add(i)
for i in range(50):
    walls.add(i)
for i in range(49, 2500, 50):
    walls.add(i)
for i in range(2450, 2500):
    walls.add(i)
for i in range(266, 666, 50):
    walls.add(i)
for i in range(240, 640, 50):
    walls.add(i)
for i in range(975, 1275, 50):
    walls.add(i)
for i in range(2116, 2140):
    walls.add(i)

f_pos = [i for i in range(2500) if i not in walls]

actions = {
    0: -50,  # up
    1: 50,   # down
    2: -1,   # left
    3: 1     # right
}

Q = np.zeros((2500, 4))

gamma = 0.9
alpha = 0.1
epsilon = 0.8

def is_valid_position(pos):
    return (0 <= pos < 2500) and (pos not in walls)

def get_local_region(sp, fp, radius=5):
    sp_row, sp_col = divmod(sp, 50)
    fp_row, fp_col = divmod(fp, 50)

    min_row = max(0, min(sp_row, fp_row) - radius)
    max_row = min(49, max(sp_row, fp_row) + radius)
    min_col = max(0, min(sp_col, fp_col) - radius)
    max_col = min(49, max(sp_col, fp_col) + radius)

    indices = []
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            idx = r * 50 + c
            if idx not in walls:
                indices.append(idx)
    return indices

# Initialize position and goal
sp = np.random.choice(f_pos)
fp = np.random.choice(f_pos)

# Draw walls once
for i in walls:
    step_dict[i][:, :, 0] = 255  # Blue = wall

step_dict[sp][:, :, 1] = 255  # Agent (green)
step_dict[fp][:, :, 2] = 255  # Goal (red)

while True:
    # --- Training loop inside main loop ---
    local_region = get_local_region(sp, fp, radius = 5)
    for _ in range(5000):  # Fewer steps, only for local region
        train_sp = random.choice(local_region)
        train_fp = fp

        if random.random() < epsilon:
            action = random.choice(list(actions.keys()))
            next_train_sp = train_sp + actions[action]
        else:
            action = np.argmax(Q[train_sp])
            next_train_sp = train_sp + actions[action]

        if not is_valid_position(next_train_sp):
            reward = -100
            next_train_sp = train_sp
        elif next_train_sp == train_fp:
            reward = 100
        else:
            reward = 0

        max_next_q = np.max(Q[next_train_sp])
        #Q[train_sp, action] = Q[train_sp, action] + alpha * (reward + gamma * max_next_q - Q[train_sp, action])
        Q[train_sp, action] = reward + gamma * max_next_q

    # --- After training update, make one move in actual env ---
    cv.imshow('img', img)
    key = cv.waitKey(100)

    # Choose best action (no exploration)
    action = np.argmax(Q[sp])
    next_sp = sp + actions[action]

    if not is_valid_position(next_sp):
        print("Hit wall or out of bounds! Game Over.")
        break

    # Clear old agent position and redraw everything
    step_dict[sp][:, :, :] = 0
    for w in walls:
        step_dict[w][:, :, 0] = 255
    step_dict[fp][:, :, 2] = 255

    sp = next_sp
    step_dict[sp][:, :, 1] = 255  # Draw agent

    if sp == fp:
        print("Goal reached!")
        fp = np.random.choice(f_pos)
        step_dict[fp][:, :, 2] = 255
        step_dict[sp][:, :, 1] = 255
        Q = np.zeros((2500, 4))

    if key == 27:  # ESC to quit
        break

cv.destroyAllWindows()