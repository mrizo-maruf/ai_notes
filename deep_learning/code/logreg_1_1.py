"""Section 1.1: NumPy logistic regression and reproducible figures.
Run from anywhere: python path/to/deep_learning/code/logreg_1_1.py
Dependencies: numpy, matplotlib. All data are synthetic, seed 7.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parents[1] / 'figures' / 'logreg_1_1'

def sigmoid(z):
    """Avoid overflow at large negative logits."""
    z = np.asarray(z, dtype=float)
    return np.exp(-np.logaddexp(0.0, -z))

def objective_gradient(X, y, w, lam=0.0):
    """Mean BCE plus lam/2 times squared slopes; column 0 is the bias."""
    z = X @ w
    loss = np.logaddexp(0.0, (1.0 - 2.0 * y) * z).mean()
    reg = w.copy()
    reg[0] = 0.0
    return (loss + lam / 2 * (reg @ reg), X.T @ (sigmoid(z) - y) / len(y) + lam * reg)

def fit(X, y, lam=0.02, steps=1800):
    w = np.zeros(X.shape[1])
    bound = np.linalg.norm(X, 2) ** 2 / (4 * len(y)) + lam
    rate = 1.0 / bound
    history = []
    for _ in range(steps):
        loss, grad = objective_gradient(X, y, w, lam)
        history.append(loss)
        w -= rate * grad
    history.append(objective_gradient(X, y, w, lam)[0])
    return (w, np.asarray(history))

def checks():
    """Verify gradients, curvature, extreme logits, and descent numerically."""
    X = np.array([[1.0, -1.0, 0.5], [1.0, 0.3, -2.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    w = np.array([0.2, -0.4, 0.7])
    lam = 0.03
    _, analytic = objective_gradient(X, y, w, lam)
    eps = 1e-06
    numeric = np.array([(objective_gradient(X, y, w + eps * e, lam)[0] - objective_gradient(X, y, w - eps * e, lam)[0]) / (2 * eps) for e in np.eye(3)])
    np.testing.assert_allclose(analytic, numeric, rtol=1e-06, atol=1e-08)
    p = sigmoid(X @ w)
    H = X.T @ ((p * (1 - p))[:, None] * X) / len(y)
    assert np.linalg.eigvalsh(H).min() >= -1e-12
    extreme = np.array([-1000.0, 1000.0])
    assert np.isfinite(sigmoid(extreme)).all()
    loss, _ = objective_gradient(np.eye(2), np.array([1.0, 0.0]), extreme)
    assert np.isfinite(loss) and loss == 1000.0
    assert np.isclose(sigmoid(0), 0.5)
    np.testing.assert_allclose(sigmoid(np.array([-2.0, 0.0, 2.0])), 1 - sigmoid(np.array([2.0, 0.0, -2.0])))
    _, hist = fit(X, y)
    assert np.max(np.diff(hist)) <= 1e-12
    return float(np.max(np.abs(analytic - numeric)))

def save(fig, name):
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)

def figures():
    """Build seven figures and run the seeded synthetic grasp experiment."""
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.constrained_layout.use': True, 'pdf.fonttype': 42})
    blue, orange = ('#235789', '#b54c1c')
    z = np.linspace(-8, 8, 401)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for c in [0.5, 1.0, 3.0]:
        ax[0].plot(z, sigmoid(c * z), label=f'scale c={c:g}')
    ax[0].axvline(0, color='gray', ls=':')
    ax[0].axhline(0.5, color='gray', ls=':')
    ax[0].set(xlabel='Unscaled score z', ylabel='Predicted probability', title='Same boundary; different confidence')
    ax[0].legend()
    xx, yy = np.meshgrid(np.linspace(-1, 4, 150), np.linspace(-1, 4, 150))
    zz = xx + yy - 3
    im = ax[1].contourf(xx, yy, sigmoid(zz), levels=np.linspace(0, 1, 11), cmap='Blues')
    cs = ax[1].contour(xx, yy, sigmoid(zz), levels=[0.1, 0.5, 0.9], colors='black')
    ax[1].clabel(cs, fmt='p=%.1f')
    ax[1].set(xlabel='$x_1$', ylabel='$x_2$', title='$z=-3+x_1+x_2$', aspect='equal')
    fig.colorbar(im, ax=ax[1], label='p(Y=1 | x)')
    save(fig, 'geometry')
    p = np.linspace(0.001, 0.999, 500)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(p, -np.log(p), label='y=1', color=blue)
    ax[0].plot(p, -np.log1p(-p), label='y=0', color=orange)
    ax[0].set(xlabel='Predicted p', ylabel='Loss (nats)', title='Confident errors are expensive', ylim=(0, 7))
    ax[0].legend()
    ax[1].plot(z, np.logaddexp(0, -z), label='y=1', color=blue)
    ax[1].plot(z, np.logaddexp(0, z), label='y=0', color=orange)
    ax[1].set(xlabel='Logit z', ylabel='Loss (nats)', title='The same loss in score coordinates')
    ax[1].legend()
    save(fig, 'bce')
    q = 0.7
    risk = -q * np.log(p) - (1 - q) * np.log1p(-p)
    entropy = -q * np.log(q) - (1 - q) * np.log(1 - q)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(p, risk, color=blue)
    ax[0].axvline(q, color=orange, ls='--', label='True q=0.7')
    ax[0].axhline(entropy, color='gray', ls=':')
    ax[0].set(xlabel='Reported probability p', ylabel='Expected BCE', title='Honest probability minimizes expected loss', ylim=(0, 3))
    ax[0].legend()
    ax[1].plot(p, -p * np.log(p) - (1 - p) * np.log1p(-p), color=blue)
    ax[1].set(xlabel='Bernoulli probability p', ylabel='Entropy (nats)', title='Maximum predictive entropy at p=0.5')
    save(fig, 'probability_meaning')
    grid = np.linspace(-2, 2, 200)
    xx, yy = np.meshgrid(grid, grid)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    score = 4 * (xx ** 2 + yy ** 2 - 1)
    im = ax[0].contourf(xx, yy, sigmoid(score), levels=np.linspace(0, 1, 11), cmap='Blues')
    ax[0].contour(xx, yy, score, levels=[0], colors='black')
    ax[0].set(xlabel='$x_1$', ylabel='$x_2$', title='Circular boundary in input space', aspect='equal')
    phi = np.linspace(0, 4, 100)
    ax[1].plot(phi, 1 - phi, color='black')
    ax[1].set(xlim=(0, 4), ylim=(0, 4), xlabel='$\\phi_1=x_1^2$', ylabel='$\\phi_2=x_2^2$', title='Straight boundary in feature space')
    ax[1].fill_between(phi, np.maximum(0, 1 - phi), 4, color=blue, alpha=0.2)
    save(fig, 'nonlinear_features')
    # A nonseparable dataset gives a finite optimum for this loss surface.
    x = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    X = np.column_stack([np.ones(6), x])
    b, a = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-2, 5, 100))
    losses = np.logaddexp(0, (1 - 2 * y)[None, None, :] * (b[:, :, None] + a[:, :, None] * x)).mean(axis=2)
    w = np.array([-2.5, -1.5])
    path = [w.copy()]
    for _ in range(100):
        w -= 0.5 * objective_gradient(X, y, w)[1]
        path.append(w.copy())
    path = np.array(path)
    fig = plt.figure(figsize=(10, 3.8))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(b, a, losses, cmap='Blues', linewidth=0, rstride=3, cstride=3, alpha=0.9)
    ax.set(xlabel='Bias b', ylabel='Slope a', zlabel='Mean BCE', title='Convex surface for fixed features')
    ax.view_init(25, -65)
    ax = fig.add_subplot(122)
    cs = ax.contour(b, a, losses, levels=[0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0])
    ax.clabel(cs, fontsize=8)
    ax.plot(path[:, 0], path[:, 1], '.-', color=orange, ms=2)
    ax.set(xlabel='Bias b', ylabel='Slope a', title='Full-batch gradient descent')
    save(fig, 'optimization')
    # Synthetic grasp trials: physical features, then Bernoulli labels.
    rng = np.random.default_rng(7)
    n = 1200
    raw = np.column_stack([rng.uniform(0, 20, n), rng.uniform(0, 40, n)])
    true_p = sigmoid(3 - 0.25 * raw[:, 0] - 0.08 * raw[:, 1])
    labels = rng.binomial(1, true_p).astype(float)
    order = rng.permutation(n)
    tr, va, te = (order[:720], order[720:960], order[960:])
    # Fit preprocessing on training data only.
    mu = raw[tr].mean(axis=0)
    sd = raw[tr].std(axis=0)
    design = lambda r: np.column_stack([np.ones(len(r)), (r - mu) / sd])
    w, hist = fit(design(raw[tr]), labels[tr], lam=0.02)
    pred = sigmoid(design(raw[te]) @ w)
    valpred = sigmoid(design(raw[va]) @ w)
    test_bce = objective_gradient(design(raw[te]), labels[te], w)[0]
    acc = np.mean((pred >= 0.5) == labels[te])
    cost = lambda pp, yy, t: np.mean(np.where(pp >= t, 8 * (1 - yy), yy))
    # Select the empirical threshold on validation data, never on test data.
    candidates = np.linspace(0.05, 0.95, 91)
    selected = float(candidates[np.argmin([cost(valpred, labels[va], t) for t in candidates])])
    e, angle = np.meshgrid(np.linspace(0, 20, 100), np.linspace(0, 40, 100))
    plane = np.column_stack([e.ravel(), angle.ravel()])
    pr = sigmoid(design(plane) @ w).reshape(e.shape)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    im = ax[0].contourf(e, angle, pr, levels=np.linspace(0, 1, 11), cmap='Blues')
    cs = ax[0].contour(e, angle, pr, levels=[0.5, 8 / 9], colors=['black', orange])
    ax[0].clabel(cs, fmt='p=%.3f')
    ax[0].set(xlabel='Centering error (mm)', ylabel='Angle error (degrees)', title='Fitted grasp-success probability')
    fig.colorbar(im, ax=ax[0])
    ax[1].plot(hist, color=blue)
    ax[1].set(xlabel='Gradient update', ylabel='Regularized training objective', title='Deterministic NumPy training')
    save(fig, 'robot_grasp')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    edges = np.linspace(0, 1, 6)
    bins = np.digitize(pred, edges[1:-1])
    counts = []
    for i in range(5):
        mask = bins == i
        counts.append(int(mask.sum()))
        if mask.any():
            ax[0].scatter(pred[mask].mean(), labels[te][mask].mean(), s=20 + mask.sum(), color=blue)
            ax[0].annotate(str(mask.sum()), (pred[mask].mean(), labels[te][mask].mean()), xytext=(5, 6), textcoords='offset points', fontsize=8)
    ax[0].plot([0, 1], [0, 1], '--', color='gray')
    ax[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted probability in bin', ylabel='Observed success fraction', title='Test reliability (numbers = bin counts)')
    thresholds = [0.5, 8 / 9, selected]
    costs = [cost(pred, labels[te], t) for t in thresholds]
    ax[1].bar(['0.5', '8/9', f'Validation\n{selected:.2f}'], costs, color=[blue, orange, 'gray'])
    ax[1].set(ylabel='Mean test decision cost', title='False success costs 8; missed success costs 1')
    save(fig, 'evaluation')
    summary = {'seed': 7, 'split_sizes': [len(tr), len(va), len(te)], 'train_mean': mu.tolist(), 'train_std': sd.tolist(), 'weights_standardized': w.tolist(), 'test_bce': float(test_bce), 'test_accuracy_at_0_5': float(acc), 'validation_threshold': selected, 'test_costs': dict(zip(['0.5', '8/9', 'validation_selected'], map(float, costs))), 'reliability_bin_counts': counts, 'gradient_check_max_error': checks()}
    (OUT / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    checks()
    figures()
