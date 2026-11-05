"""Activation shapes, curve construction, real Iris fits, and depth.
Run from the repository root: python deep_learning/code/activations_1_6_1_7.py
Training and gradients use NumPy only; math.erf supplies the exact GELU CDF.
Iris CSV was exported from sklearn.datasets.load_iris; sklearn is not required
for reproduction. Model choices/epochs are fixed, not selected on test data.
"""
from pathlib import Path
import json, math, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'/'activations_1_6_1_7'

def sig(z): return np.exp(-np.logaddexp(0.,-z))
def relu(z): return np.maximum(z,0.)
def phi(z): return np.exp(-z*z/2)/np.sqrt(2*np.pi)
def Phi(z): return .5*(1+np.vectorize(math.erf)(z/np.sqrt(2)))
def softmax(z):
    a=z-z.max(axis=1,keepdims=True)
    p=np.exp(a); return p/p.sum(axis=1,keepdims=True)

def activation(name,z):
    if name=='sigmoid': return sig(z),sig(z)*(1-sig(z))
    if name=='tanh': return np.tanh(z),1-np.tanh(z)**2
    if name=='linear': return z,np.ones_like(z)
    if name=='ReLU': return relu(z),(z>0).astype(float)
    if name=='PReLU (a=0.2)': return np.where(z>=0,z,.2*z),np.where(z>=0,1.,.2)
    if name=='ELU (a=1)': return np.where(z>=0,z,np.expm1(z)),np.where(z>=0,1.,np.exp(z))
    if name=='GELU': return z*Phi(z),Phi(z)+z*phi(z)
    if name=='Swish (beta=1)': return z*sig(z),sig(z)+z*sig(z)*(1-sig(z))
    raise ValueError(name)

def forward(X,p,kind):
    z=X@p['W1'].T+p['b1']
    h=relu(z) if kind=='regression' else np.tanh(z)
    o=h@p['W2'].T+p['b2']
    return z,h,o

def objective(X,y,p,kind,lam):
    z,h,o=forward(X,p,kind);N=len(X)
    if kind=='regression':
        e=o-y;data=.5*np.mean(e*e);do=e/N
    else:
        prob=softmax(o)
        m=o.max(axis=1,keepdims=True)
        logsum=m[:,0]+np.log(np.exp(o-m).sum(axis=1))
        data=np.mean(logsum-o[np.arange(N),y])
        do=prob.copy();do[np.arange(N),y]-=1;do/=N
    loss=data+.5*lam*(np.sum(p['W1']**2)+np.sum(p['W2']**2))
    dh=do@p['W2']
    dz=dh*((z>0) if kind=='regression' else (1-h*h))
    grad={'W2':do.T@h+lam*p['W2'],'b2':do.sum(axis=0),
          'W1':dz.T@X+lam*p['W1'],'b1':dz.sum(axis=0)}
    return loss,grad

def train(X,y,kind,seed,steps,lam):
    rng=np.random.default_rng(seed); d=X.shape[1];k=6;c=1 if kind=='regression' else 3
    p={'W1':rng.normal(0,.7,size=(k,d)), 'b1':np.linspace(-.8,.8,k),
       'W2':rng.normal(0,.3,size=(c,k)), 'b2':np.zeros(c)}
    m={a:np.zeros_like(v) for a,v in p.items()};v={a:np.zeros_like(v) for a,v in p.items()}
    history=[]
    for t in range(1,steps+1):
        loss,g=objective(X,y,p,kind,lam)
        for a in p:
            m[a]=.9*m[a]+.1*g[a];v[a]=.999*v[a]+.001*g[a]**2
            p[a]-=.01*(m[a]/(1-.9**t))/(np.sqrt(v[a]/(1-.999**t))+1e-8)
        if t==1 or t%100==0:history.append([t,float(loss)])
    return p,history

def gradient_checks():
    rng=np.random.default_rng(91);X=rng.normal(size=(5,2))
    worst={}
    for kind in ['regression','classification']:
        c=1 if kind=='regression' else 3
        p={'W1':rng.normal(size=(3,2)),'b1':rng.normal(size=3),
           'W2':rng.normal(size=(c,3)),'b2':rng.normal(size=c)}
        y=rng.normal(size=(5,1)) if kind=='regression' else np.array([0,1,2,1,0])
        _,g=objective(X,y,p,kind,.001);err=0
        for name,a in p.items():
            for index in np.ndindex(a.shape):
                orig=a[index];a[index]=orig+1e-6;lp=objective(X,y,p,kind,.001)[0]
                a[index]=orig-1e-6;lm=objective(X,y,p,kind,.001)[0];a[index]=orig
                err=max(err,abs((lp-lm)/2e-6-g[name][index]))
        assert err<1e-6;worst[kind]=err
    for name in ['sigmoid','tanh','linear','ReLU','PReLU (a=0.2)','ELU (a=1)','GELU','Swish (beta=1)']:
        z=np.array([-3.,-.7,.4,2.1]);f,g=activation(name,z)
        fd=(activation(name,z+1e-6)[0]-activation(name,z-1e-6)[0])/2e-6
        np.testing.assert_allclose(fd,g,atol=1e-8)
    # Output cross-entropy gradient p - y, including softmax cross-coordinate effects.
    logits=np.array([[1.,-2.,3.]]);p=softmax(logits)
    np.testing.assert_allclose(softmax(logits+900),p)
    assert np.isclose(p.sum(),1)
    return worst

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)

def basic_figures():
    z=np.linspace(-4,4,500)
    names=['sigmoid','tanh','linear','ReLU','PReLU (a=0.2)','ELU (a=1)','GELU','Swish (beta=1)']
    fig,axes=plt.subplots(2,4,figsize=(11,5),layout='constrained')
    for ax,name in zip(axes.ravel(),names):
        f,g=activation(name,z);ax.plot(z,f,label='activation');ax.plot(z,g,'--',label='derivative')
        ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5);ax.set_title(name);ax.set_xlabel('preactivation z')
    axes[0,0].legend(fontsize=8);save(fig,'activation_atlas')
    x=np.linspace(0,4,501)
    C=np.array([relu(x),-2*relu(x-1),3*relu(x-2),-2*relu(x-3)]).T
    f=C.sum(axis=1)
    np.testing.assert_allclose(np.interp([0,1,2,3,4],x,f),[0,1,0,2,2])
    fig,ax=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
    for j in range(4):
        ax[0].plot(x,C[:,j],label=f'contribution {j+1}')
        ax[1].plot(x,C[:,:j+1].sum(axis=1),label=f'first {j+1} units')
    ax[2].plot(x,f,color='black',lw=2);ax[2].scatter([0,1,2,3,4],[0,1,0,2,2])
    for a,t in zip(ax,['Weighted ReLU hinges','Partial sums','Final function']):a.set(xlabel='x',ylabel='output',title=t)
    ax[0].legend(fontsize=7);ax[1].legend(fontsize=7);save(fig,'hinge_sum')
    fig,ax=plt.subplots(1,2,figsize=(10,3.3),layout='constrained')
    s=12
    bumps=sig(s*(x-1))-sig(s*(x-2))
    ax[0].plot(x,sig(s*(x-1)),label='sigmoid at 1');ax[0].plot(x,-sig(s*(x-2)),label='negative sigmoid at 2');ax[0].plot(x,bumps,'k',lw=2,label='sum: soft window')
    ax[0].set(xlabel='x',title='Differences of sigmoids make local bumps');ax[0].legend(fontsize=8)
    q=np.linspace(0,1,501)
    for n in [2,4,8]:
        knots=np.linspace(0,1,n+1);values=knots**2
        ax[1].plot(q,np.interp(q,knots,values),label=f'{n} intervals')
    ax[1].plot(q,q*q,'k--',label='target x²');ax[1].set(xlabel='x',title='ReLU interpolation approaches a smooth curve');ax[1].legend(fontsize=8)
    save(fig,'bumps_and_approximation')
    fig,ax=plt.subplots(1,2,figsize=(10,3.2),layout='constrained')
    z=np.linspace(-10,10,500);p=sig(z)
    ax[0].plot(z,np.logaddexp(0.,-z),label='BCE with sigmoid (y=1)');ax[0].plot(z,.5*(p-1)**2,label='half squared error with sigmoid');ax[0].set(xlabel='logit z',ylabel='loss',title='Different likelihoods, different penalties');ax[0].legend(fontsize=8)
    ax[1].plot(z,p-1,label='BCE: p - 1');ax[1].plot(z,(p-1)*p*(1-p),label='Squared error: (p-1)p(1-p)');ax[1].set(xlabel='logit z',ylabel='derivative with respect to z',title='Confidently wrong: BCE retains a large signal');ax[1].legend(fontsize=8)
    save(fig,'loss_gradients')
    fig,ax=plt.subplots(1,3,figsize=(11,2.7),layout='constrained');x=np.linspace(0,1,4097)
    def tent(t):return 2*relu(t)-4*relu(t-.5)+2*relu(t-1)
    t=x.copy()
    for depth in range(1,4):
        t=tent(t);ax[depth-1].plot(x,t);ax[depth-1].set(xlabel='x',ylim=(-.1,1.1),title=f'{depth} composition(s): {2**depth} affine pieces')
        for knot in np.linspace(0,1,2**depth+1):ax[depth-1].axvline(knot,color='gray',lw=.3,alpha=.4)
    save(fig,'depth_folding')


def serial(p):return {k:v.tolist() for k,v in p.items()}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    checks=gradient_checks();basic_figures()
    raw=np.loadtxt(ROOT/'data'/'iris.csv',delimiter=',',skiprows=1)
    species=raw[:,4].astype(int);rng=np.random.default_rng(2026)
    train_idx=[];test_idx=[]
    for c in range(3):
        idx=rng.permutation(np.flatnonzero(species==c));train_idx.extend(idx[:40]);test_idx.extend(idx[40:])
    tr=np.array(train_idx);te=np.array(test_idx)
    x=raw[:,2:3];y=raw[:,3:4]
    mx=x[tr].mean(axis=0);sx=x[tr].std(axis=0);my=y[tr].mean(axis=0);sy=y[tr].std(axis=0)
    X=(x-mx)/sx;Y=(y-my)/sy
    rp,rhist=train(X[tr],Y[tr],'regression',11,5000,1e-4)
    pred=my+sy*forward(X,rp,'regression')[2]
    A=rp['W1'][:,0]/sx[0];B=rp['b1']-rp['W1'][:,0]*mx[0]/sx[0]
    V=sy[0]*rp['W2'][0];c0=float(my[0]+sy[0]*rp['b2'][0])
    np.testing.assert_allclose(c0+relu(x*A+B)@V,pred[:,0],atol=1e-12)
    coef=np.linalg.lstsq(np.c_[np.ones(len(tr)),x[tr]],y[tr],rcond=None)[0]
    linear=np.c_[np.ones(len(x)),x]@coef
    regression={'architecture':[1,6,1],'hidden':'ReLU','output':'linear','seed':11,'steps':5000,'learning_rate':.01,'l2_weight_penalty':1e-4,
      'x_mean':mx.tolist(),'x_std':sx.tolist(),'y_mean':my.tolist(),'y_std':sy.tolist(),'standardized_weights':serial(rp),
      'raw_formula':{'A':A.tolist(),'B':B.tolist(),'V':V.tolist(),'c':c0},
      'train_rmse_cm':float(np.sqrt(np.mean((pred[tr]-y[tr])**2))),'test_rmse_cm':float(np.sqrt(np.mean((pred[te]-y[te])**2))),
      'linear_baseline_test_rmse_cm':float(np.sqrt(np.mean((linear[te]-y[te])**2))),
      'constant_baseline_test_rmse_cm':float(np.sqrt(np.mean((my-y[te])**2))),'linear_baseline_coefficients':coef.tolist(),
      'train_history':rhist}
    grid=np.linspace(.8,7,500);parts=relu(grid[:,None]*A+B)*V
    fig,ax=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
    for j in range(6):ax[0].plot(grid,parts[:,j],label=f'unit {j+1}')
    ax[0].set(xlabel='petal length (cm)',ylabel='contribution (cm)',title='Actual learned contributions');ax[0].legend(fontsize=7,ncol=2)
    for j in range(6):ax[1].plot(grid,c0+parts[:,:j+1].sum(axis=1),label=f'first {j+1}')
    ax[1].set(xlabel='petal length (cm)',ylabel='partial prediction (cm)',title='Bias plus partial sums');ax[1].legend(fontsize=7,ncol=2)
    ax[2].scatter(x[tr],y[tr],s=12,alpha=.4,label='train');ax[2].scatter(x[te],y[te],s=28,marker='x',label='held-out test')
    ax[2].plot(grid,c0+parts.sum(axis=1),'k',label='trained 1-6-1 ReLU');ax[2].plot(grid,coef[0,0]+coef[1,0]*grid,'--',label='linear baseline')
    ax[2].set(xlabel='petal length (cm)',ylabel='petal width (cm)',title='One feature, continuous target');ax[2].legend(fontsize=7)
    save(fig,'iris_regression_decomposition')
    cx=raw[:,[0,3]];cm=cx[tr].mean(axis=0);cs=cx[tr].std(axis=0);CX=(cx-cm)/cs
    cp,chist=train(CX[tr],species[tr],'classification',19,4000,1e-3)
    logits=forward(CX,cp,'classification')[2];prob=softmax(logits);labels=prob.argmax(axis=1)
    classification={'architecture':[2,6,3],'features':['sepal_length_cm','petal_width_cm'],'hidden':'tanh','output':'softmax',
      'seed':19,'steps':4000,'learning_rate':.01,'l2_weight_penalty':1e-3,'x_mean':cm.tolist(),'x_std':cs.tolist(),'weights':serial(cp),
      'train_accuracy':float(np.mean(labels[tr]==species[tr])),'test_accuracy':float(np.mean(labels[te]==species[te])),
      'test_cross_entropy':float(-np.log(prob[te,species[te]]).mean()),'test_confusion_matrix':[[int(np.sum((species[te]==a)&(labels[te]==b))) for b in range(3)] for a in range(3)],
      'example_row':int(te[0]),'example_input':cx[te[0]].tolist(),'example_hidden':forward(CX[te[:1]],cp,'classification')[1][0].tolist(),
      'example_logits':logits[te[0]].tolist(),'example_probabilities':prob[te[0]].tolist(),'train_history':chist}
    aa,bb=np.meshgrid(np.linspace(4,8.2,180),np.linspace(0,2.8,160));gg=np.c_[aa.ravel(),bb.ravel()]
    pp=softmax(forward((gg-cm)/cs,cp,'classification')[2]);hh=forward((gg-cm)/cs,cp,'classification')[1]
    fig,ax=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    ax[0].contourf(aa,bb,pp.argmax(axis=1).reshape(aa.shape),levels=[-.5,.5,1.5,2.5],cmap='Pastel1')
    colors=['#1f77b4','#d97900','#207b46'];names=['setosa','versicolor','virginica']
    for c in range(3):
        idx=te[species[te]==c];ax[0].scatter(cx[idx,0],cx[idx,1],c=colors[c],edgecolors='black',label=names[c],s=40)
    ax[0].set(xlabel='sepal length (cm)',ylabel='petal width (cm)',title='Trained 2-6-3 classifier; test points');ax[0].legend(fontsize=8)
    v=np.linspace(0,2.8,300);slice_x=np.c_[np.full_like(v,cm[0]),v];sp=softmax(forward((slice_x-cm)/cs,cp,'classification')[2])
    for c in range(3):ax[1].plot(v,sp[:,c],color=colors[c],label=names[c])
    ax[1].set(xlabel='petal width (cm)',ylabel='class probability',title=f'Slice at sepal length = {cm[0]:.3f} cm',ylim=(-.03,1.03));ax[1].legend(fontsize=8)
    save(fig,'iris_classification')
    fig,ax=plt.subplots(2,3,figsize=(10,5),layout='constrained')
    for j,a in enumerate(ax.ravel()):
        a.contourf(aa,bb,hh[:,j].reshape(aa.shape),levels=np.linspace(-1,1,15),cmap='coolwarm')
        a.set(xlabel='sepal length (cm)',ylabel='petal width (cm)',title=f'Learned tanh feature {j+1}')
    save(fig,'iris_hidden_features')
    fig,ax=plt.subplots(1,2,figsize=(9,3),layout='constrained')
    for a,h,title in zip(ax,[rhist,chist],['Regression objective','Classification objective']):
        h=np.array(h);a.plot(h[:,0],h[:,1]);a.set(xlabel='Adam step',ylabel='training objective incl. L2',title=title)
    save(fig,'iris_training')
    result={'data_source':'CSV exported from sklearn.datasets.load_iris (corrected copy); UCI Iris, DOI 10.24432/C56C76',
       'csv_sha256':hashlib.sha256((ROOT/'data'/'iris.csv').read_bytes()).hexdigest(),
       'split_seed':2026,'train_rows_zero_based':tr.tolist(),'test_rows_zero_based':te.tolist(),
       'preprocessing':'train-only means and population standard deviations; float64',
       'optimizer':'NumPy full-batch Adam, beta1=.9 beta2=.999 epsilon=1e-8; no early stopping/test selection',
       'gradient_check_max_absolute_errors':checks,'regression':regression,'classification':classification}
    (OUT/'trained_models.json').write_text(json.dumps(result,indent=2)+'\n')
    # Generated TeX tables give real coefficients without manual transcription.
    lines=['\\begin{center}\\begin{tabular}{rrrrr}','\\toprule','Unit $j$ & $A_j$ & $B_j$ & $V_j$ & Knot $-B_j/A_j$\\\\','\\midrule']
    for j in range(6):
        knot = f'{-B[j]/A[j]:.5f}' if abs(A[j])>1e-5 else 'inactive'
        lines.append(f'{j+1} & {A[j]:.5f} & {B[j]:.5f} & {V[j]:.5f} & {knot}\\\\')
    lines+=['\\bottomrule','\\end{tabular}\\end{center}',f'Output bias in centimetres: $c={c0:.6f}$.']
    (OUT/'regression_weights.tex').write_text('\n'.join(lines)+'\n')
    lines=['\\begin{center}\\begin{tabular}{rrrrrrr}','\\toprule','Unit & $w_{j1}$ & $w_{j2}$ & $b_j$ & $v_{0j}$ & $v_{1j}$ & $v_{2j}$\\\\','\\midrule']
    for j in range(6):lines.append(' & '.join([str(j+1)]+[f'{a:.5f}' for a in [*cp['W1'][j],cp['b1'][j],*cp['W2'][:,j]]])+'\\\\')
    lines+=['\\bottomrule','\\end{tabular}\\end{center}', 'Output biases (class order 0, 1, 2): $('+', '.join(f'{a:.5f}' for a in cp['b2'])+')$.']
    (OUT/'classifier_weights.tex').write_text('\n'.join(lines)+'\n')
    summary={k:regression[k] for k in ['train_rmse_cm','test_rmse_cm','linear_baseline_test_rmse_cm','constant_baseline_test_rmse_cm']}
    summary.update({k:classification[k] for k in ['train_accuracy','test_accuracy','test_cross_entropy','test_confusion_matrix','example_row','example_input','example_hidden','example_logits','example_probabilities']})
    summary.update({'reg_x_mean':mx.tolist(),'reg_x_std':sx.tolist(),'reg_y_mean':my.tolist(),'reg_y_std':sy.tolist(),'class_x_mean':cm.tolist(),'class_x_std':cs.tolist(),'gradient_checks':checks})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
