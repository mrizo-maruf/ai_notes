"""Fixed-design sine regression: reference solution and figure generator."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "figures" / "bias_variance"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(17)
sigma, B, N = 0.15, 400, 12
x = np.linspace(0, 1, N)
grid = np.linspace(0, 1, 201)
truth = np.sin(2 * np.pi * grid)
y = np.sin(2 * np.pi * x)[:, None] + sigma * rng.normal(size=(N, B))


def design(z, degree):
    # Scaling inputs improves conditioning without changing the polynomial span.
    return np.polynomial.polynomial.polyvander(2 * z - 1, degree)


def fit(degree):
    X, Q = design(x, degree), design(grid, degree)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return (Q @ coef).T, np.mean((X @ coef - y) ** 2)


plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
for ax, degree in zip(axes, [0, 3, 11]):
    predictions, _ = fit(degree)
    ax.plot(grid, truth, "k--", label="True mean")
    ax.plot(grid, predictions[0], "k-", label="Fitted model")
    ax.scatter(x, y[:, 0], c="k", s=14, label="Training labels")
    ax.set(xlabel="Input x", title=f"Degree {degree}", ylim=(-2, 2))
axes[0].set_ylabel("Target / prediction")
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fits.pdf")
plt.close(fig)

rows = []
for degree in [0, 3, 9]:
    P, train = fit(degree)
    average = P.mean(axis=0)
    bias2 = (average - truth) ** 2
    variance = ((P - average) ** 2).mean(axis=0)
    mean_error = ((P - truth) ** 2).mean(axis=0)
    np.testing.assert_allclose(mean_error, bias2 + variance, atol=1e-12)
    # Average 40 independent test-noise realizations per fitted prediction.
    measured = np.mean([
        np.mean((P - truth - sigma * rng.normal(size=P.shape)) ** 2)
        for _ in range(40)
    ])
    rows.append(dict(degree=degree, train=float(train),
                     bias2=float(bias2.mean()), variance=float(variance.mean()),
                     noise=sigma ** 2,
                     expected=float(mean_error.mean() + sigma ** 2),
                     noisy_test=float(measured)))
assert np.all(np.diff([r["train"] for r in rows]) <= 1e-12)
F = np.array([1., 2., 3.])
Y = np.array([1.5, 2.5])
np.testing.assert_allclose(np.mean((F[:, None] - Y) ** 2), 11 / 12)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
degrees = [r["degree"] for r in rows]
for key, style, label in [("bias2", "ko-", "Bias squared"),
                           ("variance", "ks--", "Prediction variance"),
                           ("noise", "k^:", "Noise")]:
    axes[0].plot(degrees, [r[key] for r in rows], style, label=label)
for key, style, label in [("expected", "ko-", "Expected test MSE"),
                           ("train", "ks--", "Training MSE")]:
    axes[1].plot(degrees, [r[key] for r in rows], style, label=label)
for ax in axes:
    ax.set(xlabel="Polynomial degree", ylabel="Squared target units",
           yscale="log", xticks=degrees)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUT / "components.pdf")
plt.close(fig)

report = dict(seed=17, repetitions=B, N=N, sigma=sigma,
              design="fixed equally spaced inputs; fresh independent label noise",
              evaluation="mean over 201 uniform grid locations", results=rows)
(OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
header = r"""\begin{center}\small
\begin{tabular}{rrrrrr}
\toprule Degree & Training MSE & Bias$^2$ & Variance & Expected test & Noisy test\\
\midrule
"""
body = "".join(
    f"{r['degree']} & {r['train']:.5f} & {r['bias2']:.5f} & "
    f"{r['variance']:.5f} & {r['expected']:.5f} & {r['noisy_test']:.5f}"
    + r"\\" + "\n" for r in rows)
(OUT / "results_table.tex").write_text(
    header + body + "\\bottomrule\n\\end{tabular}\n\\end{center}\n")
print(json.dumps(report, indent=2))
