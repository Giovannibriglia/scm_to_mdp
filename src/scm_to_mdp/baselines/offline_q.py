"""Offline deep Q-learning baselines: offline DQN and offline CQL (model-free, no guarantee).

Both learn only from the logged transitions (the same Dataset as every other method) and never
interact with the environment: no new rollouts, no early stopping or tuning on the simulator.

    offline DQN   DQN (Mnih et al., Nature 518:529-533, 2015) trained on a fixed buffer, as studied by
                  Agarwal, Schuurmans, Norouzi, "An Optimistic Perspective on Offline Reinforcement
                  Learning", ICML 2020 (https://arxiv.org/abs/1907.04543). No pessimism: the max of the
                  target can pick actions the data never show in that state.
    offline CQL   Conservative Q-Learning, CQL(H) for discrete actions (Kumar, Zhou, Tucker, Levine,
                  NeurIPS 2020, https://arxiv.org/abs/2006.04779): the same loss plus
                      alpha * mean_s [ logsumexp_a Q(s, a) - Q(s, a_data) ],
                  which pushes Q down on actions the data do not support.

Choices (fixed for every environment and grid, before any run):
    input       the state variables, each one-hot, concatenated (the network sees the factored state)
    output      Q(s, a_k) for the K action vectors seen in the data (the argmax is over those)
    network     MLP, 2 hidden layers of `hidden` units, ReLU; PyTorch on the CPU, one thread, seeded
    targets     Double DQN: y = r + gamma * Q_target(s', argmax_a Q(s', a)), 0 after a terminal state;
                Huber loss; the target network is copied every `target_every` steps
    training    Adam (lr), `steps` gradient steps of `batch` transitions drawn uniformly from the data;
                the last network is kept (offline there is no fair way to pick a checkpoint)
    reward      R[s, a] + r_next[s'] from the known Reward, as for every other method
    CQL alpha   0.1 (tool/methods.py, CQL_ALPHA). Chosen on a pilot before the grids (seed 0; SysAdmin
                N = 3, 6; Sepsis, 2 vitals; Chemistry, 5 objects; uniform and expert loggers): alpha = 1
                (the default of d3rlpy's DiscreteCQL) pulls Q towards equal values over the actions
                when the data are uniform, and the greedy policy reached only 71-85% of V*_obs there
                (0.1: 87-99%); with expert data 0.1, 0.3 and 1 all reached 99-100%. 50,000 steps
                instead of 20,000 changed nothing for the better.
    checked     against d3rlpy 2.8.1 (DoubleDQN, DiscreteCQL) with the same network, steps and alpha
                on SysAdmin N = 4 and Sepsis: the same Q levels (e.g. mean max Q 26.35 vs 26.40) and
                the same failures (offline DQN on Sepsis with expert data)

The learned Q is evaluated on every observed state: the model is the table Q [S, K]; the policy is
its argmax and the "promise" its max, which is an estimate (DQN) or a pessimistic estimate (CQL),
not a certified bound. A tabular offline DQN would converge to the optimum of the empirical MDP, the
flat MDP baseline: the point of these baselines is a network that generalises across states.
"""

from dataclasses import dataclass

import numpy as np

from scm_to_mdp.core import Dataset, Solution


@dataclass(frozen=True)
class QTable:
    """A learned Q over the observed states and the data's action vectors, with training notes."""

    Q: np.ndarray  # [S, K]
    info: dict


def features(dataset: Dataset) -> np.ndarray:
    """[S, sum of cards] the one-hot encoding of every state, variable by variable."""
    cards = dataset.state_cards
    out = np.zeros((len(dataset.B), sum(cards)), dtype=np.float32)
    offset = 0
    for i, c in enumerate(cards):
        out[np.arange(len(dataset.B)), offset + dataset.B[:, i]] = 1.0
        offset += c
    return out


def learn(
    dataset: Dataset,
    gamma: float,
    cql_alpha: float = 0.0,
    steps: int = 20_000,
    batch: int = 256,
    hidden: int = 256,
    lr: float = 3e-4,
    target_every: int = 1_000,
    seed: int = 0,
) -> QTable:
    """Train offline DQN (cql_alpha = 0) or offline CQL (cql_alpha > 0) on the dataset."""
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    df = dataset.df
    s = dataset.state_index(df[dataset.state_vars].values)
    s_next = dataset.state_index(df[dataset.next_vars].values)
    k = dataset.action_index()
    reward = dataset.reward
    r = reward.R[s, k] + reward.r_next[s_next]
    goes_on = (~reward.terminal[s_next]).astype(np.float32)
    X = torch.from_numpy(features(dataset))
    K = len(dataset.actions)
    data = {
        "s": torch.from_numpy(s.astype(np.int64)),
        "k": torch.from_numpy(k.astype(np.int64)),
        "r": torch.from_numpy(r.astype(np.float32)),
        "s_next": torch.from_numpy(s_next.astype(np.int64)),
        "goes_on": torch.from_numpy(goes_on),
    }

    def network():
        return nn.Sequential(
            nn.Linear(X.shape[1], hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, K),
        )

    q, target = network(), network()
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    huber = nn.SmoothL1Loss()
    n = len(s)
    losses = []
    for step in range(steps):
        i = torch.from_numpy(rng.integers(0, n, size=min(batch, n)))
        si, ki, ri, s2, cont = (data[c][i] for c in ("s", "k", "r", "s_next", "goes_on"))
        with torch.no_grad():
            best = q(X[s2]).argmax(1, keepdim=True)  # Double DQN: the online net chooses
            y = ri + gamma * cont * target(X[s2]).gather(1, best).squeeze(1)  # the target net values
        logits = q(X[si])
        q_data = logits.gather(1, ki[:, None]).squeeze(1)
        loss = huber(q_data, y)
        if cql_alpha > 0:
            loss = loss + cql_alpha * (torch.logsumexp(logits, 1) - q_data).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % target_every == 0:
            target.load_state_dict(q.state_dict())
        if step >= steps - 1_000:
            losses.append(float(loss))
    with torch.no_grad():
        Q = q(X).double().numpy()
    info = {"final_loss": float(np.mean(losses)), "transitions": n, "steps": steps}
    return QTable(Q, info)


def greedy(model: QTable) -> Solution:
    """The greedy policy of Q and its estimated value (not a bound)."""
    return Solution(model.Q.max(1), model.Q.argmax(1))
