"""Reproducible forward-pass experiments for sections 1.4 and 1.5.
Run: python deep_learning/code/mlp_1_4_1_5.py
NumPy implements all models; Matplotlib draws the explanatory figures.
Examples are constructed, not measured robotics benchmarks.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'figures' / 'mlp_1_4_1_5'

def relu(z):
    return np.maximum(z, 0.0)

def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))

def dense_loop(X, W, b):
    """W stores one neuron's weights per row; explicit scalar reference."""
    B, d = X.shape
    k = W.shape[0]
    Z = np.empty((B, k))
    for n in range(B):
        for j in range(k):
            total = b[j]
            for i in range(d):
                total += W[j, i] * X[n, i]
            Z[n, j] = total
    return Z

def forward(X, weights, biases, keep=False):
    """ReLU hidden layers and sigmoid output. Batch rows are independent."""
    H = X
    cache = []
    for layer, (W, b) in enumerate(zip(weights, biases)):
        if W.shape[1] != H.shape[1] or b.shape != (W.shape[0],):
            raise ValueError('Incompatible dense-layer shapes')
        Z = H @ W.T + b
        H = sigmoid(Z) if layer == len(weights)-1 else relu(Z)
        if keep:
            cache.append((Z, H))
    return H, cache

def costs(widths, batch):
    """Scalar counts, not seconds or measured allocator/GPU memory.
    All weights are resident. Full cache stores Z and H separately.
    Inference workspace model: layer input + preactivation + activation.
    Input is counted once and remains resident. No gradients/optimizer.
    """
    d = widths[0]
    edges = sum(a*b for a, b in zip(widths[:-1], widths[1:]))
    biases = sum(widths[1:])
    params = edges + biases
    cache = batch*d + 2*batch*biases
    # At layer 1 its input is already included in the resident X.
    live = max(batch*(2*widths[1]),
               max((batch*(widths[l-1]+2*widths[l])
                    for l in range(2, len(widths))), default=0))
    return dict(weight_products=batch*edges, parameters=params,
                all_cached_scalars=params+cache,
                inference_scalars=params+batch*d+live)

def xor_model(X):
    W = np.array([[1., -1.], [-1., 1.]])
    H = relu(X @ W.T)
    logits = 10.0 * (H.sum(axis=1) - 0.5)
    return H, sigmoid(logits)

def checks():
    rng = np.random.default_rng(45)
    # Asymmetric sizes catch accidental transposes and bias-axis errors.
    X = rng.normal(size=(7, 3))
    W = rng.normal(size=(5, 3)); b = rng.normal(size=5)
    Z = X @ W.T + b
    np.testing.assert_allclose(Z, dense_loop(X, W, b), atol=1e-12)
    W_slide = W.T
    Z_columns = W_slide.T @ X.T + b[:, None]
    np.testing.assert_allclose(Z, Z_columns.T)
    np.testing.assert_allclose(Z, X @ W.T + np.ones((7, 1)) @ b[None, :])
    weights = [rng.normal(size=(5, 3)), rng.normal(size=(2, 5)), rng.normal(size=(1, 2))]
    biases = [rng.normal(size=n) for n in [5, 2, 1]]
    pred, cache = forward(X, weights, biases, True)
    one_by_one = np.vstack([forward(x[None, :], weights, biases)[0] for x in X])
    np.testing.assert_allclose(pred, one_by_one, atol=1e-12)
    assert pred.shape == (7, 1)
    assert [h.shape for _, h in cache] == [(7, 5), (7, 2), (7, 1)]
    # Two affine maps really do collapse, including the propagated bias.
    U, V = weights[:2]; a, c = biases[:2]
    stacked = (X @ U.T + a) @ V.T + c
    collapsed = X @ (V @ U).T + (V @ a + c)
    np.testing.assert_allclose(stacked, collapsed, atol=1e-12)
    corners = np.array([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
    H, p = xor_model(corners)
    np.testing.assert_array_equal(p >= .5, [False, True, True, False])
    Xn = np.array([[1.,2.,3.], [-1.,0.,2.]])
    Wn = np.array([[1.,-1.,2.], [0.,3.,-1.]])
    bn = np.array([.5,-.5])
    np.testing.assert_allclose(Xn @ Wn.T + bn, [[5.5,2.5],[3.5,-2.5]])
    stat = costs([3,5,2,1], 7)
    actual_params = sum(w.size+b.size for w,b in zip(weights,biases))
    actual_cache = X.size + sum(z.size+h.size for z,h in cache)
    assert stat['parameters'] == actual_params
    assert stat['all_cached_scalars'] == actual_params + actual_cache
    # Every uniform-width formula includes M=1 without a fictitious k*k layer.
    for M in [1,2,5]:
        d,k,B=3,4,7
        stat = costs([d]+[k]*M, B)
        assert stat['weight_products'] == B*(d*k+(M-1)*k*k)
        assert stat['parameters'] == d*k+(M-1)*k*k+M*k
    return dict(checks='passed: scalar/vectorized, column/row, bias, batch independence, affine collapse, XOR, shapes, memory counts',
                xor_probabilities=p.tolist(), worked_matrix_output=(Xn@Wn.T+bn).tolist(),
                robot_single=costs([12,32,16,1],1),
                robot_batch=costs([12,32,16,1],256))

def save(fig, name):
    fig.savefig(OUT/(name+'.pdf'), bbox_inches='tight',
                metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)

def figures():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    v=np.linspace(-.3,1.3,220); a,b=np.meshgrid(v,v)
    pts=np.c_[a.ravel(),b.ravel()]
    _,p=xor_model(pts)
    corners=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    H,pc=xor_model(corners); labels=np.array([0,1,1,0])
    fig,ax=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    im=ax[0].contourf(a,b,p.reshape(a.shape),levels=np.linspace(0,1,15),cmap='Blues')
    ax[0].contour(a,b,p.reshape(a.shape),levels=[.5],colors='black')
    ax[0].scatter(*corners.T,c=labels,cmap='coolwarm',vmin=0,vmax=1,edgecolors='black',s=90)
    ax[0].set(xlabel='$x_1$',ylabel='$x_2$',title='Input space: two boundary segments')
    for xy,lab in zip(corners,labels): ax[0].annotate(str(lab),xy,xytext=(7,7),textcoords='offset points')
    ax[1].plot([0,.5],[.5,0],color='black',label='$h_1+h_2=0.5$')
    ax[1].scatter(*H.T,c=labels,cmap='coolwarm',vmin=0,vmax=1,edgecolors='black',s=100)
    ax[1].annotate('(0,0) and (1,1)',(0,0),xytext=(8,15),textcoords='offset points')
    ax[1].set(xlim=(-.1,1.2),ylim=(-.1,1.2),xlabel=r'$h_1=\max(0,x_1-x_2)$',
              ylabel=r'$h_2=\max(0,x_2-x_1)$',title='Hidden space: one linear separator')
    ax[1].legend(loc='upper right',fontsize=9)
    save(fig,'xor_features')
    fig,axs=plt.subplots(1,4,figsize=(11,3),layout='constrained')
    matrices=[np.array([[1,2,3],[-1,0,2]]), np.array([[1,0],[-1,3],[2,-1]]),
              np.array([[.5,-.5],[.5,-.5]]), np.array([[5.5,2.5],[3.5,-2.5]])]
    titles=['X: (2 examples, 3 features)','$W^T$: (3 features, 2 neurons)',
            'Bias broadcast: (2, 2)','Z: (2 examples, 2 neurons)']
    for ax,A,title in zip(axs,matrices,titles):
        ax.imshow(np.ones_like(A),cmap='Blues',vmin=0,vmax=3)
        for (i,j),value in np.ndenumerate(A): ax.text(j,i,f'{value:g}',ha='center',va='center',fontsize=13)
        ax.set(xticks=np.arange(A.shape[1]),yticks=np.arange(A.shape[0]),title=title)
        ax.tick_params(length=0); ax.set_title(title,fontsize=9)
    fig.suptitle('Z = X @ W.T + b: each output cell is one dot product plus one bias')
    save(fig,'matrix_example')
    fig,ax=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    B=32; d=128; M=4; k=np.arange(8,257,8)
    ax[0].plot(k,B*d*k,label='Input layer: B d k')
    ax[0].plot(k,B*(M-1)*k*k,label='Remaining layers: B (M-1) k²')
    ax[0].set(xlabel='Uniform width k',ylabel='Weighted feature products',title='Widening hidden layers has quadratic cost')
    ax[0].legend(fontsize=8)
    batches=np.arange(1,513,8); widths=[128]+[64]*4
    params=costs(widths,1)['parameters']
    ax[1].plot(batches,np.full_like(batches,params)*4/1024,label='Parameters (shared)')
    ax[1].plot(batches,[costs(widths,int(bb))['all_cached_scalars']*4/1024 for bb in batches],label='Parameters + all Z/H + input')
    ax[1].plot(batches,[costs(widths,int(bb))['inference_scalars']*4/1024 for bb in batches],label='Parameters + live inference buffers')
    ax[1].set(xlabel='Batch size B',ylabel='Model scalar storage at float32 (KiB)',title='Activation storage grows with the batch')
    ax[1].legend(fontsize=8)
    save(fig,'complexity_scaling')
    # Fixed width, increasing depth exposes why live and retained activations differ.
    depths=np.arange(1,21); B=512;d=k=64
    retained=[B*d+2*B*m*k for m in depths]
    live=[costs([d]+[k]*int(m),B)['inference_scalars']-costs([d]+[k]*int(m),B)['parameters'] for m in depths]
    fig,ax=plt.subplots(figsize=(8,3),layout='constrained')
    ax.plot(depths,np.array(retained)*4/1024,label='Input + all preactivations/activations')
    ax.plot(depths,np.array(live)*4/1024,label='Input + current live buffers')
    ax.set(xlabel='Number of computed layers M',ylabel='Activation/input storage (KiB)',
           title='Depth accumulates cached activations; inference can reuse buffers',xticks=[1,5,10,15,20])
    ax.legend(fontsize=8)
    save(fig,'activation_memory')

if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    result=checks()
    figures()
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
