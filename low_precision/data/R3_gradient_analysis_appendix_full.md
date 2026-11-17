# Gradient Analysis of Routing Replay

论文附录完整草稿。原始推导 `MOE概率模型的推导.md` 保持不变。

本稿以选中集合内 softmax 归一化为前提，仍需与实际训练实现核对。结论是 R3 不保证保留训练端自由路由目标的梯度，而不是每次 replay 都产生非零偏差。下面的三专家构造是新增证明例子，不是实验结果。

## English LaTeX

放在已有的 `\appendix` 后。

```latex
\section{Gradient Analysis of Routing Replay}
\label{R3_analysis}

This section examines how R3~\citep{ma2025r3} changes the gradients
used for policy optimization. We first derive the local gradients of an MoE layer.
We then identify the signals excluded by replay and compare the
replay objective with the free-routing objective.

\subsection{Setup and Assumptions}

We consider a sparsely gated MoE layer~\citep{shazeer2017sparsely}.
For one token at this layer, we omit the token and layer indices
for clarity. Let $h$ be the layer input, and let
\begin{equation}
z = W_r h + b_r,
\qquad
S^{\mathrm{cur}} = \operatorname{TopK}(z).
\end{equation}
Free routing uses the current expert set $S^{\mathrm{cur}}$.
R3~\citep{ma2025r3} instead uses the expert set
$S^{\mathrm{roll}}$ recorded during rollout.

We assume that replay fixes the expert identities but recomputes
the gate weights from the current router logits. For a nonempty
set $S$, the weights are normalized within the selected set:
\begin{equation}
g_e^S =
\frac{\exp(z_e)}{\sum_{i\in S}\exp(z_i)},
\qquad e\in S.
\label{eq:r3-normalized-gate}
\end{equation}
The MoE output is
\begin{equation}
o^S = \sum_{e\in S} g_e^S E_e(h;\psi_e),
\label{eq:r3-moe-output}
\end{equation}
where $\psi_e$ denotes the parameters of expert $e$.

The following derivation concerns gradients through this MoE
output. Expert parameters are separate, and expert outputs do
not depend directly on the router logits. Auxiliary router
losses and other parameter uses are excluded from the local
derivatives. We also assume that selected experts are evaluated
without token dropping.

The selected set is held fixed during backpropagation. For free
routing, this is the usual local derivative where the top-$k$
set does not change. At selection boundaries, the output can
be nondifferentiable or discontinuous. Neither free routing nor
R3 obtains an ordinary gradient through the discrete selection
switch.

\subsection{Gradients for a Fixed Expert Set}

Let $\mathcal{L}^S$ denote the downstream loss evaluated using
set $S$. Define
\begin{equation}
\delta^S = \frac{\partial \mathcal{L}^S}{\partial o^S},
\qquad
a_e^S = (\delta^S)^\top E_e(h;\psi_e).
\end{equation}
The superscript $S$ matters: changing the selected experts can
change both the output and the downstream gradient.

For $e,j\in S$, differentiating
Equation~\ref{eq:r3-normalized-gate} gives
\begin{equation}
\frac{\partial g_e^S}{\partial z_j}
=
g_e^S\bigl(\mathbf{1}[e=j]-g_j^S\bigr).
\end{equation}
Therefore,
\begin{equation}
\begin{aligned}
\frac{\partial o^S}{\partial z_j}
&=
\sum_{e\in S}
E_e(h;\psi_e)
\frac{\partial g_e^S}{\partial z_j}\\
&=
g_j^S\bigl(E_j(h;\psi_j)-o^S\bigr),
\qquad j\in S.
\end{aligned}
\end{equation}
For $j\notin S$, neither the normalized gates nor the output
depends on $z_j$. Thus,
\begin{equation}
r_j^S
\equiv
\frac{\partial \mathcal{L}^S}{\partial z_j}
=
\begin{cases}
g_j^S
\left(
a_j^S-\sum_{e\in S}g_e^S a_e^S
\right),
& j\in S,\\
0,
& j\notin S.
\end{cases}
\label{eq:r3-logit-gradient}
\end{equation}

For the linear router, the corresponding local contributions are
\begin{equation}
\left.\nabla_{W_r}\mathcal{L}^S\right|_{\mathrm{local}}
=
r^S h^\top,
\qquad
\left.\nabla_{b_r}\mathcal{L}^S\right|_{\mathrm{local}}
=
r^S.
\end{equation}
Contributions from different tokens are summed for shared router
parameters.

Let
$J_j(h)=\partial E_j(h;\psi_j)/\partial\psi_j$
be the expert Jacobian. The direct expert gradient is
\begin{equation}
\left.\nabla_{\psi_j}\mathcal{L}^S\right|_{\mathrm{local}}
=
\begin{cases}
g_j^S J_j(h)^\top \delta^S,
& j\in S,\\
0,
& j\notin S.
\end{cases}
\label{eq:r3-direct-expert-gradient}
\end{equation}
These are contributions from one token at one layer.
An expert excluded here can still receive gradients from other
tokens.

\subsection{Learning Signals Excluded by Replay}

\paragraph{Proposition 1.}
Fix the layer input $h$ and the current model parameters.
Suppose
$j\in S^{\mathrm{cur}}\setminus S^{\mathrm{roll}}$.
The replay pass gives zero local gradients to $z_j$ and $\psi_j$.
The free-routing expert gradient is nonzero whenever
\begin{equation}
J_j(h)^\top\delta^{\mathrm{cur}}\ne 0.
\end{equation}
Its local logit gradient is nonzero whenever
\begin{equation}
(\delta^{\mathrm{cur}})^\top
\left(E_j(h;\psi_j)-o^{\mathrm{cur}}\right)\ne 0.
\end{equation}

\paragraph{Proof.}
Since $j\notin S^{\mathrm{roll}}$, the replay output does not
depend directly on $z_j$ or $\psi_j$.
Equations~\ref{eq:r3-logit-gradient}
and~\ref{eq:r3-direct-expert-gradient}
therefore give zero local gradients.
Under free routing, $j$ is selected and $g_j^{\mathrm{cur}}>0$
for finite logits. The two stated conditions then imply
nonzero expert and logit gradients, respectively.
\hfill$\square$

This result identifies the signals excluded by replay.
It does not imply that all router gradients vanish.
Experts in $S^{\mathrm{roll}}$ can still receive gate and
expert gradients. Experts shared by both sets can also receive
different gradients because the gate normalization and downstream
gradient depend on the selected set.

For normalized top-$1$ routing, the sole gate equals one, so
the local logit gradient is zero under both methods.
The expert-gradient distinction still applies.
Thus, missing expert gradients should not be confused with
missing gradients through the discrete selection itself.

The fixed-input comparison isolates the effect of changing one
expert set. In a multilayer model, earlier replay decisions can
also change the inputs to later layers. The formulas above apply
to each actual forward pass, but the two passes need not share
the same $h$, $z$, or $\delta$.

\subsection{Difference from the Free-Routing Objective}

Let
$\mathcal{R}^{\mathrm{cur}}_\theta(x,y)$ denote the complete
routing assignments under the current training policy, and let
$\mathcal{R}^{\mathrm{roll}}$ be the assignments recorded during
rollout. Fix a distribution $\mathcal{D}$ over training examples
$(x,y,\mathcal{R}^{\mathrm{roll}})$ during the update.
Define
\begin{equation}
\begin{aligned}
J_{\mathrm{cur}}(\theta)
&=
\mathbb{E}_{\mathcal{D}}
\left[
\ell\left(
\theta;x,y,\mathcal{R}^{\mathrm{cur}}_\theta(x,y)
\right)
\right],\\
J_{\mathrm{R3}}(\theta)
&=
\mathbb{E}_{\mathcal{D}}
\left[
\ell\left(
\theta;x,y,\mathcal{R}^{\mathrm{roll}}
\right)
\right].
\end{aligned}
\label{eq:r3-objectives}
\end{equation}
Both objectives use the same data distribution and loss rule.
Only the routing rule differs. Here, $\ell$ is the training
loss evaluated on the specified route. Fixed advantages and
other training targets are held constant during differentiation.

At differentiable parameter values, assume that differentiation
can be interchanged with expectation. Define
\begin{equation}
b_{\mathrm{replay}}(\theta)
=
\nabla_\theta J_{\mathrm{R3}}(\theta)
-
\nabla_\theta J_{\mathrm{cur}}(\theta).
\label{eq:r3-gradient-bias}
\end{equation}
A stochastic gradient that is unbiased for $J_{\mathrm{R3}}$
has bias $b_{\mathrm{replay}}$ when used to estimate the
gradient of $J_{\mathrm{cur}}$.

In general, a replayed set that differs from the current set
evaluates a different route, so $b_{\mathrm{replay}}(\theta)$
need not be zero. R3 can therefore exclude learning signals
available under free routing and produce a different policy
gradient. This difference need not be nonzero for every example,
and gradients may cancel in expectation. The analysis identifies
a limitation of routing replay, rather than establishing that
replay always reduces performance.
```

## 中文对照与核查重点

这版附录的完整论证分为五步。

### 1. 明确比较条件

在同一 token、同一层、固定输入下，自由路由选择当前 top-k 专家，R3 使用 rollout 记录的专家。两者都用当前 logits，在选中集合内重新归一化。只分析该层输出带来的局部梯度，不包含辅助损失、token dropping 或其他参数使用路径。

### 2. 推导固定集合梯度

对选中专家，

\[
\frac{\partial\mathcal L^S}{\partial z_j}
=
g_j^S(\delta^S)^\top(E_j-o^S),
\qquad
\left.\nabla_{\psi_j}\mathcal L^S\right|_{\mathrm{local}}
=
g_j^S J_j^\top\delta^S.
\]

未选专家的这两项局部梯度为零。不同路由下的上游梯度 $\delta^S$ 也可能不同，不能直接当作同一个量。

### 3. 证明缺失信号

对当前路由选中、rollout 路由未选中的专家，R3 排除了它从该 token 获得的直接梯度。自由路由下的梯度在给定非零条件时存在。这不意味着整个专家的总梯度为零，也不意味着 R3 冻结了全部 router 参数。

### 4. 用具体构造证明梯度差异可以非零

三个专家中，当前选择 $\{1,2\}$，rollout 选择 $\{1,3\}$。采用一个明确的策略梯度损失，可以算出专家 2 及对应 logit 在自由路由下的梯度非零，在 replay 下为零。再选择只包含该样本的数据分布，就能证明这种差异也可以在期望中保留。因此，R3 一般不保证给出自由路由目标的无偏梯度。

### 5. 连接 RDC，但不扩大结论

RDC 不替换保留样本的训练路由，因此保留这些样本在自由路由下的梯度计算。过滤仍会改变参与更新的样本，不能据此宣称 RDC 对原始目标无偏，也不能直接把 routing score 等同于梯度误差。

与精简版相比，三专家构造补上了“梯度可能不一致”的证明。它是新增的证明例子，不是原始文档已有的实验或实证结果。
