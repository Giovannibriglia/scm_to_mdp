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
    network     MLP, 2 hidden layers of `hidden` units, ReLU, float32; PyTorch, seeded. The weights are
                initialised on the CPU and the batches drawn by numpy, so every device starts from the
                same network and sees the same batches; the CPU run is bit-for-bit reproducible, a GPU
                run agrees with it up to float rounding (utils/compute.py chooses the device)
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

import warnings
from dataclasses import dataclass

import numpy as np

from scm_to_mdp.core import Dataset, Solution

CHUNK = 1_000  # training steps whose batches are drawn at once
WARMUP = 20  # ordinary CUDA steps before the training step is recorded as a CUDA graph
SYNC_EVERY = 32  # graph replays queued at most: then the CPU sleeps until the GPU caught up


def _capture(torch, update, opt, i):
    """Record update(i) as a CUDA graph: (graph, its input tensor, its loss), or Nones on failure."""
    try:
        static_i = i.clone()
        opt.zero_grad(set_to_none=True)  # the graph allocates the gradients and overwrites them
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_loss = update(static_i)
        return g, static_i, static_loss
    except Exception as err:  # noqa: BLE001 - a GPU without graph support trains step by step
        warnings.warn(f"CUDA graph capture failed, training step by step: {err!r}", stacklevel=2)
        opt.zero_grad(set_to_none=True)
        return None, None, None


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
    device: str | None = None,
    cuda_graph: bool = True,
) -> QTable:
    """Train offline DQN (cql_alpha = 0) or offline CQL (cql_alpha > 0) on the dataset.
    device: cpu, cuda, cuda:N, mps; None = SCM_TO_MDP_DEVICE, else the CPU (utils/compute.py)."""
    import torch
    from torch import nn

    from scm_to_mdp.utils.compute import training_device, training_threads

    dev = training_device(device)
    torch.set_num_threads(training_threads())
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    df = dataset.df
    s = dataset.state_index(df[dataset.state_vars].values)
    s_next = dataset.state_index(df[dataset.next_vars].values)
    k = dataset.action_index()
    reward = dataset.reward
    r = reward.R[s, k] + reward.r_next[s_next]
    goes_on = (~reward.terminal[s_next]).astype(np.float32)
    X = torch.from_numpy(features(dataset)).to(dev)
    K = len(dataset.actions)
    data = {
        "s": torch.from_numpy(s.astype(np.int64)).to(dev),
        "k": torch.from_numpy(k.astype(np.int64)).to(dev),
        "r": torch.from_numpy(r.astype(np.float32)).to(dev),
        "s_next": torch.from_numpy(s_next.astype(np.int64)).to(dev),
        "goes_on": torch.from_numpy(goes_on).to(dev),
    }

    def network():
        return nn.Sequential(
            nn.Linear(X.shape[1], hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, K),
        )

    q, target = network().to(dev), network().to(dev)  # initialised on the CPU: the same on every device
    target.load_state_dict(q.state_dict())
    graph = cuda_graph and dev.type == "cuda"
    opt = torch.optim.Adam(q.parameters(), lr=lr, capturable=graph)  # capturable: its step lives on the GPU
    huber = nn.SmoothL1Loss()
    n = len(s)
    b = min(batch, n)

    def update(i):
        """One gradient step on the transitions i (grads must be None or zero before)."""
        si, ki, ri, s2, cont = (data[c][i] for c in ("s", "k", "r", "s_next", "goes_on"))
        with torch.no_grad():
            best = q(X[s2]).argmax(1, keepdim=True)  # Double DQN: the online net chooses
            y = ri + gamma * cont * target(X[s2]).gather(1, best).squeeze(1)  # the target net values
        logits = q(X[si])
        q_data = logits.gather(1, ki[:, None]).squeeze(1)
        loss = huber(q_data, y)
        if cql_alpha > 0:
            loss = loss + cql_alpha * (torch.logsumexp(logits, 1) - q_data).mean()
        loss.backward()
        opt.step()
        return loss.detach()

    # CUDA graph: after WARMUP ordinary steps, one step is recorded and then replayed with one launch
    # per step (the same kernels on the same batches; it removes the per-kernel launch cost that
    # dominates for networks this small). Any failure falls back to ordinary steps.
    replay, static_i, static_loss = None, None, None
    caught_up = torch.cuda.Event(blocking=True) if graph else None  # waits by sleeping, not spinning
    losses = []  # the last 1,000 losses, kept on the device (no synchronisation per step)
    for step in range(steps):
        if step % CHUNK == 0:  # the batches of the next CHUNK steps in one draw (the same numbers)
            block = torch.from_numpy(rng.integers(0, n, size=(min(CHUNK, steps - step), b))).to(dev)
        i = block[step % CHUNK]
        if graph and step == WARMUP:
            replay, static_i, static_loss = _capture(torch, update, opt, i)
            graph = replay is not None
        if replay is not None:
            static_i.copy_(i)
            replay.replay()
            loss = static_loss.clone()
            if step % SYNC_EVERY == 0:  # a full launch queue would busy-wait a CPU core per training
                caught_up.record()
                caught_up.synchronize()
        else:
            opt.zero_grad()
            loss = update(i)
        if (step + 1) % target_every == 0:
            target.load_state_dict(q.state_dict())  # in place: the recorded graph sees the new weights
        if step >= steps - 1_000:
            losses.append(loss)
    with torch.no_grad():
        Q = q(X).cpu().double().numpy()
    final = float(torch.stack(losses).cpu().double().mean()) if losses else float("nan")
    info = {"final_loss": final, "transitions": n, "steps": steps, "device": str(dev)}
    info["cuda_graph"] = replay is not None
    return QTable(Q, info)


def greedy(model: QTable) -> Solution:
    """The greedy policy of Q and its estimated value (not a bound)."""
    return Solution(model.Q.max(1), model.Q.argmax(1))
