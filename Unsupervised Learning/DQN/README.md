# Deep Q-Network (DQN)

> Q-Learning with a neural network instead of a table.

[⬅ Back to main README](../../README.md)

---

## Definition

DQN replaces the Q-table with a neural network that estimates `Q(s, a)`. This lets the agent handle large or continuous state spaces where a table would never fit. It is a **reinforcement learning** method, grouped here for layout convenience.

## Intuition

A table works for a small maze, but not for a video game screen with millions of possible states. A network can generalize: similar states get similar values, even if the agent has never seen that exact state.

## Formulas

**Target value**

$$y = r + \gamma\max_{a'}Q_{\theta^-}(s', a')$$

**Loss**

$$L(\theta) = \mathbb{E}\left[\left(y - Q_\theta(s,a)\right)^2\right]$$

**Techniques that keep training stable**

- **Experience replay:** store past transitions and train on random mini-batches to break correlations.
- **Target network:** a slowly updated copy of the network ($\theta^-$) used to compute targets.

## Files in this folder

- `DQN.py`: builds the Q-network, replay buffer, and training loop.

## How to run

```bash
python DQN.py
```

## Expected outcome

- A trained network whose reward per episode improves over time.
- A policy derived from `argmax Q(s, a)`.

## Tips and hyperparameters

- **Replay buffer size and batch size:** larger buffers give more stable learning.
- **Target update frequency:** update the target network every few hundred steps.
- Training can be noisy, so average reward over many episodes.

## Strengths and limitations

**Strengths:** scales to huge state spaces.

**Limitations:** sample-hungry, sensitive to hyperparameters, and can be unstable without replay and target networks.

---

[⬅ Back to main README](../../README.md)
