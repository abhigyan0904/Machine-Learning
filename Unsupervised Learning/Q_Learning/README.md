# Q-Learning

> Learn the best action in every situation by trial and error.

[⬅ Back to main README](../../README.md)

---

## Definition

Q-Learning is a **reinforcement learning** algorithm. An agent interacts with an environment, receives rewards, and stores the estimated value of every state-action pair in a table (the Q-table). It is grouped in this folder for layout convenience.

## Intuition

A robot in a maze tries moves, gets rewards for good ones and penalties for bad ones, and slowly writes down which move is best in each square. Eventually it just follows the notes.

## Formulas

**Bellman update rule**

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\right]$$

**Policy**

$$\pi(s) = \arg\max_{a}Q(s,a)$$

| Symbol | Meaning |
|---|---|
| `s`, `a` | Current state and action |
| `r` | Reward received |
| `s'` | Next state |
| `α` | Learning rate |
| `γ` | Discount factor: how much future rewards matter |

**ε-greedy exploration:** with probability `ε` take a random action, otherwise take the best known one.

## Files in this folder

- `Q_Learning.py`: trains an agent with a Q-table and prints or plots the learning progress.

## How to run

```bash
python Q_Learning.py
```

## Expected outcome

- A learned Q-table.
- Total reward per episode that rises over time.
- A policy the agent can follow to solve the task.

## Tips and hyperparameters

- **`γ` near 1:** values long-term reward. **`γ` near 0:** values immediate reward.
- **Decay `ε`** over time to shift from exploring to exploiting.
- Too high an `α` makes learning unstable.

## Strengths and limitations

**Strengths:** simple, needs no model of the environment.

**Limitations:** the table becomes impractical for large or continuous state spaces, which is what DQN solves.

---

[⬅ Back to main README](../../README.md)
