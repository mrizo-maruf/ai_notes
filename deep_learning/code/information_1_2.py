"""Entropy, cross-entropy, KL and MLE: NumPy experiments for section 1.2.
Run from the repository root: python deep_learning/code/information_1_2.py
Requires NumPy and Matplotlib. All examples are constructed, not measured data.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'figures' / 'information_1_2'
BLUE, ORANGE, GREEN = '#235789', '#b54c1c', '#34765d'


def distribution(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Expected a finite nonnegative probability vector.')
    if not np.isclose(p.sum(), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError('Probabilities must sum to one.')
    return p


def cross_entropy(p, q, base=2.0):
    p, q = distribution(p), distribution(q)
    if p.shape != q.shape:
        raise ValueError('Distributions must use the same outcome space.')
    if base <= 1 or not np.isfinite(base):
        raise ValueError('Choose a finite logarithm base greater than one.')
    active = p > 0
    if np.any(q[active] == 0):
        return float('inf')
    return float(-np.sum(p[active] * np.log(q[active])) / np.log(base))


def entropy(p, base=2.0):
    return cross_entropy(p, p, base)


def kl(p, q, base=2.0):
    p, q = distribution(p), distribution(q)
    if p.shape != q.shape:
        raise ValueError('Distributions must use the same outcome space.')
    if base <= 1 or not np.isfinite(base):
        raise ValueError('Choose a finite logarithm base greater than one.')
    active = p > 0
    if np.any(q[active] == 0):
        return float('inf')
    return float(np.sum(p[active] * (np.log(p[active]) - np.log(q[active]))) / np.log(base))


def log_normal(x, mean, std):
    return -0.5 * ((x - mean) / std)**2 - np.log(std) - 0.5*np.log(2*np.pi)


def log_target(x):
    return np.logaddexp(log_normal(x, -3.0, 0.7), log_normal(x, 3.0, 0.7)) - np.log(2)


def reverse_kl(mean, std, nodes, weights):
    """E_Q log(Q/P) via Gauss-Hermite quadrature, not a finite x-grid."""
    x = mean + np.sqrt(2)*std*nodes
    return -0.5*np.log(2*np.pi*np.e*std**2) - weights @ log_target(x)


def reverse_fit(nodes, weights):
    # Search the left half; symmetry gives an equally good right-hand fit.
    best = (float('inf'), None, None)
    for mean in np.linspace(-4, 0, 161):
        for std in np.linspace(0.35, 4.5, 167):
            value = reverse_kl(mean, std, nodes, weights)
            if value < best[0]:
                best = (value, mean, std)
    _, center, scale = best
    for mean in np.linspace(center - 0.03, center + 0.03, 61):
        for std in np.linspace(scale - 0.03, scale + 0.03, 61):
            value = reverse_kl(mean, std, nodes, weights)
            if value < best[0]:
                best = (value, mean, std)
    return best


def checks():
    p = np.array([0.5, 0.25, 0.125, 0.125])
    q = np.full(4, 0.25)
    np.testing.assert_allclose(entropy(p), 1.75)
    np.testing.assert_allclose(cross_entropy(p, q), 2.0)
    np.testing.assert_allclose(kl(p, q), 0.25)
    assert entropy([1, 0]) == 0
    assert kl([1, 0], [0, 1]) == float('inf')
    np.testing.assert_allclose(kl([1, 0], [1, 0]), 0)
    rng = np.random.default_rng(12)
    for _ in range(100):
        p, q = rng.dirichlet(np.ones(5), size=2)
        np.testing.assert_allclose(cross_entropy(p, q), entropy(p) + kl(p, q))
        assert kl(p, q) >= -1e-12
    # NLL equals cross-entropy of categorical empirical frequencies.
    labels = np.array([0, 0, 0, 1, 2])
    q = np.array([0.5, 0.3, 0.2])
    empirical = np.bincount(labels, minlength=3) / len(labels)
    np.testing.assert_allclose(-np.log(q[labels]).mean(), cross_entropy(empirical, q, np.e))
    # Conditional Bernoulli cross-entropy equals binary cross-entropy.
    y, p = np.array([0., 1., 1.]), np.array([0.2, 0.6, 0.8])
    categorical = np.mean([cross_entropy([1-yy, yy], [1-pp, pp], np.e) for yy, pp in zip(y, p)])
    np.testing.assert_allclose(categorical, -np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))
    # Reparameterizing a Gaussian changes differential entropy, not KL.
    def gaussian_kl(m1, s1, m2, s2):
        return np.log(s2/s1)+(s1*s1+(m1-m2)**2)/(2*s2*s2)-0.5
    np.testing.assert_allclose(gaussian_kl(0, 1, 2, 3), gaussian_kl(5, 100, 205, 300))


def save(fig, name):
    # Fixed metadata makes regeneration independent of wall-clock time.
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def main():
    checks()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.constrained_layout.use': True,
                         'pdf.fonttype': 42})
    x = np.linspace(0.001, 0.999, 600)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(x, -np.log2(x), color=BLUE)
    ax[0].set(xlabel='Probability of the observed event', ylabel='Surprise (bits)',
              title='Rare events carry more information')
    ax[1].plot(x, -x*np.log2(x)-(1-x)*np.log2(1-x), color=BLUE)
    ax[1].axvline(0.5, color='gray', ls=':')
    ax[1].set(xlabel='Bernoulli success probability', ylabel='Entropy (bits)',
              title='Average surprise peaks at a fair coin')
    save(fig, 'surprise_entropy')

    p = np.array([0.5, 0.25, 0.125, 0.125])
    q = np.full(4, 0.25)
    r = np.array([0.125, 0.125, 0.25, 0.5])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    positions = np.arange(4)
    for i, (prob, label, color) in enumerate([(p, 'P: source', BLUE), (q, 'Q: uniform', ORANGE), (r, 'R: mismatch', GREEN)]):
        ax[0].bar(positions + (i-1)*0.25, prob, width=0.25, label=label, color=color)
    ax[0].set(xticks=positions, xticklabels=['Clear', 'Person', 'Cart', 'Blocked'],
              ylabel='Probability', title='The source and two coding assumptions')
    ax[0].legend(fontsize=8)
    costs = [cross_entropy(p, model) for model in [p, q, r]]
    ax[1].bar(['P code', 'Q code', 'R code'], [entropy(p)]*3, color=BLUE, label='Source entropy')
    ax[1].bar(['P code', 'Q code', 'R code'], np.array(costs)-entropy(p),
              bottom=entropy(p), color=ORANGE, label='Excess = KL(P || model)')
    ax[1].set(ylabel='Expected bits per event', title='Cross-entropy = entropy + excess')
    ax[1].legend(fontsize=8)
    save(fig, 'coding')

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    true = np.array([0.8, 0.2])
    ax[0].plot(x, [kl(true, [v, 1-v], np.e) for v in x], label='KL(P || Q)', color=BLUE)
    ax[0].plot(x, [kl([v, 1-v], true, np.e) for v in x], label='KL(Q || P)', color=ORANGE)
    ax[0].set(xlabel='Q(first outcome)', ylabel='KL (nats)', title='Same optimum, different penalties', ylim=(0, 3))
    ax[0].axvline(0.8, color='gray', ls=':'); ax[0].legend()
    for p0, label, color in [(0.5, 'P = fair coin', BLUE), (0.95, 'P = biased coin', ORANGE)]:
        curve = np.array([cross_entropy([p0,1-p0], [v,1-v], np.e) for v in x])
        ax[1].plot(x, curve, label=label, color=color)
        ax[1].scatter([p0], [entropy([p0,1-p0],np.e)], color=color)
        ax[1].axhline(entropy([p0,1-p0],np.e), color=color, ls=':', alpha=0.6)
    ax[1].set(xlabel='Reported probability of first outcome', ylabel='Cross-entropy (nats)',
              title='Different sources have different loss floors', ylim=(0, 3))
    ax[1].legend(fontsize=8)
    save(fig, 'kl_and_floor')

    nodes, weights = np.polynomial.hermite.hermgauss(120)
    weights = weights / np.sqrt(np.pi)
    reverse_value, reverse_mean, reverse_std = reverse_fit(nodes, weights)
    nodes_hi, weights_hi = np.polynomial.hermite.hermgauss(240)
    weights_hi = weights_hi / np.sqrt(np.pi)
    reverse_hi = reverse_kl(reverse_mean, reverse_std, nodes_hi, weights_hi)
    assert abs(reverse_value - reverse_hi) < 1e-5
    forward_std = np.sqrt(3**2+0.7**2)
    grid = np.linspace(-8, 8, 1200)
    target = np.exp(log_target(grid))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for axis in ax:
        axis.plot(grid, target, color=BLUE, label='P: two-mode belief')
        axis.set(xlabel='Robot position (m)', ylabel='Density (1/m)', xlim=(-8,8))
    ax[0].plot(grid, np.exp(log_normal(grid, 0, forward_std)), '--', color=GREEN,
               label='Q: forward-KL optimum')
    ax[0].set_title('Forward KL: cover both plausible locations')
    ax[1].plot(grid, np.exp(log_normal(grid, reverse_mean, reverse_std)), '--', color=GREEN,
               label='Q: numerical reverse-KL fit')
    ax[1].plot(grid, np.exp(log_normal(grid, -reverse_mean, reverse_std)), ':', color=ORANGE,
               label='Symmetric alternative')
    ax[1].set_title('Reverse KL: one-mode approximation')
    for axis in ax: axis.legend(fontsize=8)
    save(fig, 'mixture_direction')

    # Objective slices make clear that these are different optimization problems.
    means = np.linspace(-4.5, 4.5, 240)
    expected_log_p = 0.5*sum(weights_hi @ log_target(mu+np.sqrt(2)*0.7*nodes_hi) for mu in [-3,3])
    h_p = -expected_log_p
    forward = 0.5*np.log(2*np.pi*forward_std**2)+(9.49+means**2)/(2*forward_std**2)-h_p
    reverse = [reverse_kl(mu, reverse_std, nodes_hi, weights_hi) for mu in means]
    fig, ax = plt.subplots(1,2,figsize=(10,3.4))
    ax[0].plot(means, forward, color=BLUE)
    ax[0].set(xlabel='Gaussian mean', ylabel='KL(P || Q) (nats)', title='Forward slice: standard deviation fixed at 3.081')
    ax[1].plot(means, reverse, color=ORANGE)
    ax[1].set(xlabel='Gaussian mean', ylabel='KL(Q || P) (nats)', title='Reverse slice: standard deviation fixed near 0.7')
    save(fig, 'mixture_objectives')

    # Bernoulli MLE: seven successes, three failures.
    likelihood = x**7 * (1-x)**3
    nll = -0.7*np.log(x)-0.3*np.log1p(-x)
    fig, ax = plt.subplots(1,2,figsize=(10,3.4))
    ax[0].plot(x, likelihood, color=BLUE)
    ax[0].axvline(0.7,color='gray',ls=':')
    ax[0].set(xlabel='Bernoulli parameter theta', ylabel='Likelihood of the observed sequence', title='Maximize the product')
    ax[1].plot(x,nll,color=BLUE,label='Empirical cross-entropy')
    ax[1].plot(x,nll-entropy([.7,.3],np.e),color=ORANGE,label='Empirical KL')
    ax[1].set(xlabel='Bernoulli parameter theta',ylabel='Nats per observation',title='The same estimator minimizes both',ylim=(0,3))
    ax[1].legend(fontsize=8)
    save(fig,'mle')

    # Grouped conditional semantic classifier: finite categorical feature values.
    counts = np.array([[80,15,5], [10,20,70]])
    empirical = counts / counts.sum(axis=1,keepdims=True)
    models = np.array([[.7,.2,.1], [.2,.3,.5]])
    weights_context = counts.sum(axis=1)/counts.sum()
    conditional_h = sum(w*entropy(pp,np.e) for w,pp in zip(weights_context,empirical))
    conditional_ce = sum(w*cross_entropy(pp,qq,np.e) for w,pp,qq in zip(weights_context,empirical,models))
    conditional_kl = sum(w*kl(pp,qq,np.e) for w,pp,qq in zip(weights_context,empirical,models))
    direct_nll = -np.sum(counts*np.log(models))/counts.sum()
    np.testing.assert_allclose(conditional_ce, direct_nll)
    np.testing.assert_allclose(conditional_ce, conditional_h+conditional_kl)
    fig,ax=plt.subplots(1,2,figsize=(10,3.4))
    for i,axis in enumerate(ax):
        axis.bar(np.arange(3)-.18,empirical[i],width=.36,color=BLUE,label='Empirical label frequency')
        axis.bar(np.arange(3)+.18,models[i],width=.36,color=ORANGE,label='Model prediction')
        axis.set(xticks=np.arange(3),xticklabels=['Free','Person','Cart'],ylim=(0,1),ylabel='Probability',title=['Open-floor context','Occluded context'][i])
        axis.legend(fontsize=8)
    save(fig,'conditional_robot')
    result={'units':'nats unless field ends in bits','coding_entropy_bits':entropy(p),
            'coding_cross_entropies_bits':costs,'bernoulli_kl_08_05':kl([.8,.2],[.5,.5],np.e),
            'bernoulli_kl_05_08':kl([.5,.5],[.8,.2],np.e),'forward_gaussian_mean':0.,
            'forward_gaussian_std':float(forward_std),'mixture_entropy':float(h_p),
            'forward_kl_at_optimum':float(0.5*np.log(2*np.pi*np.e*forward_std**2)-h_p),
            'reverse_numerical_mean':float(reverse_mean),'reverse_numerical_std':float(reverse_std),
            'reverse_kl':float(reverse_hi),'quadrature_difference':float(abs(reverse_hi-reverse_value)),
            'conditional_entropy':float(conditional_h),'conditional_cross_entropy':float(conditional_ce),
            'conditional_kl':float(conditional_kl)}
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
