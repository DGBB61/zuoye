"""按网页配色生成 MLP 实验展示图（浅色 / 深色两套）。

运行：python tools/gen_web_figures.py
输出：images/mlp/*.png  +  tools/run_output.txt
算法与 G:\\workbudy\\2026-11-19-22-54-42\\xor_mlp_fixed.py 完全一致（2-2-1, sigmoid, BCE-with-logits, Adam）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ 路径
BASE = r"G:\workbudy\2026-11-19-22-03-14"
OUT = os.path.join(BASE, "images", "mlp")
LOG = os.path.join(BASE, "tools", "run_output.txt")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ 数据
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)
LABELS = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]

# ------------------------------------------------------------------ 配色（对齐网页 CSS 变量）
THEMES = {
    "light": dict(face="#ffffff", text="#2c2c2c", sub="#666666", grid="#e8e8e8",
                  accent="#2563eb", pos="#c53030", neg="#2b6cb0", edge="#2c2c2c"),
    "dark": dict(face="#242424", text="#e0e0e0", sub="#a0a0a0", grid="#3a3a3a",
                 accent="#60a5fa", pos="#fc8181", neg="#63b3ed", edge="#e0e0e0"),
}
RC = {"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
      "axes.unicode_minus": False}
LEVELS = np.linspace(0, 1, 21)


# ------------------------------------------------------------------ 数值稳定算子
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce_with_logits(z, y):
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def forward(x, p):
    h = sigmoid(x @ p["W1"] + p["b1"])
    return h, h @ p["W2"] + p["b2"]


def predict_proba(x, p):
    return sigmoid(forward(x, p)[1])


def is_solved(p):
    return bool(np.all((predict_proba(X, p) >= 0.5).astype(float) == Y))


# ------------------------------------------------------------------ 训练（与 xor_mlp_fixed.py 一致）
@dataclass
class Config:
    lr: float = 0.1
    epochs: int = 5000
    seed: int = 42
    n_hidden: int = 2
    snap_at: tuple = (0, 50, 300, 1500, 5000)
    tol: float = 1e-3
    n_restarts: int = 8
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class Result:
    params: dict
    history: list = field(default_factory=list)
    snaps: dict = field(default_factory=dict)
    seed: int = 0
    epochs_run: int = 0
    first_solved: int | None = None
    solved: bool = False
    restarts: int = 0


def init_params(seed, n_hidden):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, 1, (2, n_hidden)), "b1": np.zeros((1, n_hidden)),
            "W2": rng.normal(0, 1, (n_hidden, 1)), "b2": np.zeros((1, 1))}


def train_once(cfg, seed):
    P = init_params(seed, cfg.n_hidden)
    m = {k: np.zeros_like(a) for k, a in P.items()}
    v = {k: np.zeros_like(a) for k, a in P.items()}

    def loss_now(p):
        return bce_with_logits(forward(X, p)[1], Y)

    snap_at = {t for t in cfg.snap_at if 0 <= t <= cfg.epochs}
    res = Result(params=P, seed=seed)
    res.history.append(loss_now(P))
    if 0 in snap_at:
        res.snaps[0] = {k: a.copy() for k, a in P.items()}
    if is_solved(P):
        res.first_solved = 0

    for t in range(1, cfg.epochs + 1):
        h, z = forward(X, P)
        dz = (sigmoid(z) - Y) / len(X)
        dhp = (dz @ P["W2"].T) * h * (1 - h)
        g = {"W2": h.T @ dz, "b2": dz.sum(0, keepdims=True),
             "W1": X.T @ dhp, "b1": dhp.sum(0, keepdims=True)}
        for k in P:
            m[k] = cfg.beta1 * m[k] + (1 - cfg.beta1) * g[k]
            v[k] = cfg.beta2 * v[k] + (1 - cfg.beta2) * g[k] ** 2
            mhat = m[k] / (1 - cfg.beta1 ** t)
            vhat = v[k] / (1 - cfg.beta2 ** t)
            P[k] -= cfg.lr * mhat / (np.sqrt(vhat) + cfg.eps)

        cur_loss = loss_now(P)
        res.history.append(cur_loss)
        if t in snap_at:
            res.snaps[t] = {k: a.copy() for k, a in P.items()}
        if res.first_solved is None and is_solved(P):
            res.first_solved = t
        if cur_loss < cfg.tol and is_solved(P):
            break

    res.epochs_run = len(res.history) - 1
    res.params = P
    res.solved = is_solved(P)
    res.snaps.setdefault(res.epochs_run, {k: a.copy() for k, a in P.items()})
    return res


def train(cfg):
    best = None
    for i in range(cfg.n_restarts):
        seed = cfg.seed if i == 0 else cfg.seed + i
        r = train_once(cfg, seed)
        r.restarts = i
        if best is None or r.history[-1] < best.history[-1]:
            best = r
        if r.solved:
            return r
    return best


# ------------------------------------------------------------------ 绘图辅助
def _prep(figsize, t):
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_facecolor(t["face"])
    ax.set_facecolor(t["face"])
    ax.tick_params(colors=t["sub"], labelsize=9)
    for s in ax.spines.values():
        s.set_color(t["grid"])
    return fig, ax


def _decorate(fig, ax, t, title, xlabel=None, ylabel=None):
    ax.set_title(title, color=t["text"], fontsize=12, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=t["sub"], fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=t["sub"], fontsize=10)
    ax.grid(alpha=0.25, color=t["grid"], linewidth=0.7)
    fig.tight_layout()


def _legend(ax, t, **kw):
    leg = ax.legend(fontsize=8, framealpha=0.92, **kw)
    leg.get_frame().set_facecolor(t["face"])
    leg.get_frame().set_edgecolor(t["grid"])
    for txt in leg.get_texts():
        txt.set_color(t["sub"])
    return leg


def _colorbar(fig, ax, cf, t, label="P(y=1)"):
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=t["sub"], labelsize=8)
    cb.set_label(label, color=t["sub"], fontsize=9)
    cb.outline.set_edgecolor(t["grid"])
    return cb


# ------------------------------------------------------------------ 各类图
def draw_boundary(p, epoch, loss_val):
    gx, gy = np.meshgrid(np.linspace(-0.2, 1.2, 220), np.linspace(-0.2, 1.2, 220))
    grid = np.c_[gx.ravel(), gy.ravel()]
    prob = predict_proba(grid, p).reshape(gx.shape)
    for name, t in THEMES.items():
        fig, ax = _prep((5.0, 4.4), t)
        cf = ax.contourf(gx, gy, prob, levels=LEVELS, cmap="RdBu_r",
                         vmin=0, vmax=1, alpha=0.9)
        ax.contour(gx, gy, prob, levels=[0.5],
                   colors=("k" if name == "light" else "w"), linewidths=1.8)
        for cls, color, marker in [(0, t["neg"], "o"), (1, t["pos"], "s")]:
            pts = X[Y.ravel() == cls]
            ax.scatter(pts[:, 0], pts[:, 1], c=color, marker=marker, s=95,
                       edgecolors=t["edge"], linewidths=1.1, zorder=5, label=f"y={cls}")
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect("equal")
        _decorate(fig, ax, t, f"决策边界 · epoch {epoch}\nloss={loss_val:.3f}", "x1", "x2")
        _legend(ax, t, loc="upper right")
        _colorbar(fig, ax, cf, t)
        fig.savefig(os.path.join(OUT, f"boundary-{epoch:04d}-{name}.png"),
                    facecolor=t["face"], bbox_inches="tight")
        plt.close(fig)


def draw_loss(res):
    hist = np.maximum(np.asarray(res.history), 1e-12)
    for name, t in THEMES.items():
        fig, ax = _prep((7.2, 3.6), t)
        ax.semilogy(hist, lw=1.5, color=t["accent"], label="BCE (with logits)")
        ax.axhline(np.log(2), ls=":", lw=1, color=t["sub"], label="ln2 ≈ 随机猜测")
        for ep in sorted(res.snaps):
            if 0 < ep <= res.epochs_run:
                ax.plot(ep, max(res.history[ep], 1e-12), "o", ms=5, color=t["pos"])
        if res.first_solved:
            ax.axvline(res.first_solved, ls="--", lw=1.2, color=t["accent"],
                       label=f"首次全对 @{res.first_solved}")
        _decorate(fig, ax, t, "训练损失曲线", "Epoch", "BCE（对数轴）")
        _legend(ax, t, loc="upper right")
        fig.savefig(os.path.join(OUT, f"loss-{name}.png"), facecolor=t["face"], bbox_inches="tight")
        plt.close(fig)


def draw_hidden(res):
    H = forward(X, res.params)[0]
    w2 = res.params["W2"].ravel()
    b2 = float(res.params["b2"].ravel()[0])
    hg1, hg2 = np.meshgrid(np.linspace(-0.05, 1.05, 220), np.linspace(-0.05, 1.05, 220))
    hp = sigmoid(hg1 * w2[0] + hg2 * w2[1] + b2)
    for name, t in THEMES.items():
        fig, ax = _prep((5.0, 4.4), t)
        ax.contourf(hg1, hg2, hp, levels=LEVELS, cmap="RdBu_r",
                    vmin=0, vmax=1, alpha=0.35)
        ax.contour(hg1, hg2, hp, levels=[0.5],
                   colors=("k" if name == "light" else "w"), linewidths=1.8)
        # 合并重合点，避免标注互相遮挡
        used, groups = [False] * 4, []
        for i in range(4):
            if used[i]:
                continue
            grp = [i]
            used[i] = True
            for j in range(i + 1, 4):
                if np.linalg.norm(H[i] - H[j]) < 0.03:
                    grp.append(j)
                    used[j] = True
            groups.append(grp)
        for grp in groups:
            a, b = H[grp[0]]
            color = t["pos"] if Y.ravel()[grp[0]] == 1 else t["neg"]
            ax.scatter(a, b, c=color, s=120, edgecolors=t["edge"], zorder=5)
            txt = "\n".join(f"{i}: {LABELS[i]}  y={int(Y.ravel()[i])}" for i in grp)
            ax.annotate(txt, (a, b), xytext=(10, 12), textcoords="offset points",
                        fontsize=8, color=t["sub"],
                        bbox=dict(boxstyle="round,pad=0.25", fc=t["face"],
                                  ec=t["grid"], alpha=0.9),
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=t["sub"]))
        ax.set_xlim(-0.12, 1.15)
        ax.set_ylim(-0.12, 1.15)
        ax.set_aspect("equal")
        _decorate(fig, ax, t, "隐藏空间：线性可分", "h1", "h2")
        fig.savefig(os.path.join(OUT, f"hidden-{name}.png"), facecolor=t["face"], bbox_inches="tight")
        plt.close(fig)


def draw_proba(res):
    proba = predict_proba(X, res.params).ravel()
    for name, t in THEMES.items():
        fig, ax = _prep((6.0, 3.6), t)
        colors = [t["pos"] if q >= 0.5 else t["neg"] for q in proba]
        bars = ax.bar(range(4), proba, color=colors, edgecolor=t["edge"],
                      linewidth=0.8, width=0.55)
        for i, (bar, q) in enumerate(zip(bars, proba)):
            ax.text(bar.get_x() + bar.get_width() / 2, min(q + 0.03, 0.96),
                    f"{q:.4f}", ha="center", fontsize=9, color=t["text"])
        ax.axhline(0.5, ls="--", lw=1.1, color=t["sub"])
        ax.set_xticks(range(4))
        ax.set_xticklabels(LABELS)
        ax.set_ylim(0, 1.05)
        _decorate(fig, ax, t, "四个样本的预测概率", "样本", "P(y=1)")
        fig.savefig(os.path.join(OUT, f"proba-{name}.png"), facecolor=t["face"], bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------ 主流程
def main():
    logs = []

    def log(s=""):
        print(s)
        logs.append(s)

    cfg = Config()
    res = train(cfg)
    proba = predict_proba(X, res.params).ravel()
    final_loss = res.history[-1]
    snaps = [(e, float(res.history[e])) for e in sorted(res.snaps) if e <= res.epochs_run]

    log("=== XOR MLP 训练结果（网页出图）===")
    log(f"seed={res.seed}")
    log(f"restarts={res.restarts}")
    log(f"epochs_run={res.epochs_run}")
    log(f"first_solved={res.first_solved}")
    log(f"final_loss={final_loss:.8f}")
    log(f"solved={res.solved}")
    log(f"n_hidden={cfg.n_hidden}")
    log(f"lr={cfg.lr}  beta1={cfg.beta1}  beta2={cfg.beta2}  eps={cfg.eps}  tol={cfg.tol}")
    log("proba=" + json.dumps({k: round(float(v), 4) for k, v in zip(LABELS, proba)},
                              ensure_ascii=False))
    log("W1=" + np.array2string(res.params["W1"], precision=3, separator=", "))
    log("b1=" + np.array2string(res.params["b1"], precision=3, separator=", "))
    log("W2=" + np.array2string(res.params["W2"], precision=3, separator=", "))
    log("b2=" + np.array2string(res.params["b2"], precision=3, separator=", "))
    log("snapshots=" + json.dumps([[int(e), round(v, 6)] for e, v in snaps]))

    with plt.rc_context(RC):
        for e, v in snaps:
            draw_boundary(res.snaps[e], e, v)
        draw_loss(res)
        draw_hidden(res)
        draw_proba(res)

    log(f"image_dir={OUT}")
    log(f"log_file={LOG}")

    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(logs) + "\n")


if __name__ == "__main__":
    main()
