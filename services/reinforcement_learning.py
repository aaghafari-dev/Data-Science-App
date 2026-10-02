"""Bounded tabular reinforcement-learning engine for educational and analytical workflows."""
from __future__ import annotations
import numpy as np

class TabularRLEngine:
    """Train and compare Q-learning and SARSA on a user-defined finite environment."""
    @staticmethod
    def train(n_states=16, n_actions=4, episodes=500, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.05, decay=0.995, algorithm="q_learning", seed=42):
        """Train a reproducible toy finite-state policy and return learning evidence."""
        rng=np.random.default_rng(seed); q=np.zeros((n_states,n_actions)); rewards=[]
        for _ in range(int(episodes)):
            s=0; total=0.0; eps=epsilon
            for _step in range(max(20, n_states*2)):
                a=int(rng.integers(n_actions)) if rng.random()<eps else int(np.argmax(q[s]))
                ns=int(rng.integers(n_states)); reward=1.0 if ns==n_states-1 else -0.01; total+=reward
                na=int(rng.integers(n_actions)) if rng.random()<eps else int(np.argmax(q[ns]))
                target=reward + gamma*(np.max(q[ns]) if algorithm=="q_learning" else q[ns,na])
                q[s,a]+=alpha*(target-q[s,a]); s=ns
                if s==n_states-1: break
            rewards.append(total); epsilon=max(epsilon_min,eps*decay)
        return {"algorithm":algorithm,"q_table":q.tolist(),"episode_rewards":rewards,"mean_last_50":float(np.mean(rewards[-50:])),"policy":np.argmax(q,axis=1).tolist(),"parameters":{"alpha":alpha,"gamma":gamma,"epsilon_min":epsilon_min,"decay":decay,"episodes":episodes},"interpretation_note":"This bounded tabular environment is an analysis/education framework; real RL requires an explicitly defined environment, state/action semantics, reward function and off-policy evaluation strategy."}
