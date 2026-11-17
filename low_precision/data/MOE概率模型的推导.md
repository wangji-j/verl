# MOE概率模型的推导

# MoE 概率模型推导

## 推导目标

在语言模型中，如果把 expert 的选择看作一个隐变量，那么 MoE 对当前目标 token 的预测概率可以写成一个关于 expert 的加权求和形式：

$p_\Theta(y_t\mid c_t)=\sum_{e=1}^{N}q_\phi(e\mid c_t)p_{\psi_e}(y_t\mid c_t,e)
$

其中，router 给出每个 expert 被选择的概率，expert 给出在该 expert 条件下生成目标 token 的概率。

---

## 基本符号定义

考虑自回归语言模型在第 $t$ 个位置预测目标 token $y_t$。

- 将当前上下文记为：

$c_t=(x,y_{<t})
$

其中：$x$表示输入 prompt 或问题；$y_{<t}$ 表示已经生成或已知的前缀 token；$y_t$表示当前位置需要预测的目标 token。

- 第 $t$ 个位置的 hidden state 记为：$h_t$

- MoE 模型中有$N$个 experts。我们把 expert 的编号记为:

$e\in{1,2,\dots,N}$

- Router 给出每个 expert 的概率：router 根据当前$c_t$或 $h_t$，输出每个 expert 被选择的概率：

$q_\phi(e\mid c_t)$

这里$\phi$是 router 的参数，通常包括 router 线性层的权重和偏置。它表示：在给定上下文的情况下，router 认为 expert $e$应该负责当前 token 的概率。从概率建模角度看，$q_\phi(e\mid c_t)$可以理解为 expert 选择的先验分布。

- Expert 给出真实 token 的条件概率：第$e$个 expert 在给定上下文和 expert 编号的条件下，对真实 token $y_t$给出的概率记为：

$p_{\psi_e}(y_t\mid c_t,e)$

这里$\psi_e$是第$e$个 expert 的参数。这个概率表示：如果当前 token 由 expert $e$负责，那么 $e$ 认为真实 token $y_t$出现的概率是多少。

---

## 简化的MoE 层对目标 token 的总概率模型

根据全概率公式，有：

${p_\Theta(y_t\mid c_t)=\sum_{e=1}^{N}q_\phi(e\mid c_t)p_{\psi_e}(y_t\mid c_t,e)}$

其中$\Theta$ 表示整个 MoE 模型的参数：$\Theta=\{\phi,\psi_1,\psi_2,\dots,\psi_N\}$

---

### 对 log probability 求梯度

训练语言模型时，单个 token 的负对数似然为：

$\mathcal{L}t=-\log p\Theta(y_t\mid c_t)$

为了推导反向传播，我们先推：

$\nabla_\Theta \log p_\Theta(y_t\mid c_t)$

根据对数函数求导公式：

$\nabla_\Theta \log f(\Theta)=\frac{\nabla_\Theta f(\Theta)}{f(\Theta)}$

有：

$\nabla_\Theta \log p_\Theta(y_t\mid c_t)=\frac{\nabla_\Theta p_\Theta(y_t\mid c_t)}{p_\Theta(y_t\mid c_t)}$

分子：

$\nabla_\Theta p_\Theta(y_t\mid c_t)=\nabla_\Theta \sum_e q_\phi(e\mid c_t)p_{\psi_e}(y_t\mid c_t,e)$

$\nabla_\Theta p_\Theta(y_t\mid c_t)=\sum_e \nabla_\Theta\left[q_\phi(e\mid c_t)p_{\psi_e}(y_t\mid c_t,e)\right]$

对每一项使用乘法求导，得到：

$\nabla_\Theta p_\Theta(y_t\mid c_t)=
\sum_e
\left[
p_{\psi_e}(y_t\mid c_t,e)\nabla_\Theta q_\phi(e\mid c_t)
+
q_\phi(e\mid c_t)\nabla_\Theta p_{\psi_e}(y_t\mid c_t,e)
\right]$

代回 log probability 的梯度公式，写成：

$\nabla_\Theta \log p_\Theta(y_t\mid c_t)
=\sum_e
\frac{
p_{\psi_e}(y_t\mid c_t,e)\nabla_\Theta q_\phi(e\mid c_t)
+
q_\phi(e\mid c_t)\nabla_\Theta p_{\psi_e}(y_t\mid c_t,e)
}{p_\Theta(y_t\mid c_t)}$

---

### 把它改写成 posterior responsibility 的形式

上一步得到：

$\nabla_\Theta \log p_\Theta(y_t\mid c_t)
=\sum_e
\frac{
p_{\psi_e}(y_t\mid c_t,e)\nabla_\Theta q_\phi(e\mid c_t)
+
q_\phi(e\mid c_t)\nabla_\Theta p_{\psi_e}(y_t\mid c_t,e)
}{p_\Theta(y_t\mid c_t)}$

接下来分别处理分子中的两项。

#### 1\. 第一项：router 概率的梯度项

第一项是：

$\frac{p_{\psi_e}(y_t\mid c_t,e)\nabla_\Theta q_\phi(e\mid c_t)}{p_\Theta(y_t\mid c_t)}$

乘除一个 $q_\phi(e\mid c_t)$：

$\frac{p_{\psi_e}(y_t\mid c_t,e)\nabla_\Theta q_\phi(e\mid c_t)}{p_\Theta(y_t\mid c_t)}
=\frac{q_\phi(e\mid c_t)p_{\psi_e}(y_t\mid c_t,e)}{p_\Theta(y_t\mid c_t)}
\frac{\nabla_\Theta q_\phi(e\mid c_t)}{q_\phi(e\mid c_t)}$

所以第一项可以写成：

$\frac{p_{\psi_e}(y_t\mid c_t,e)\nabla_\Theta q_\phi(e\mid c_t)}{p_\Theta(y_t\mid c_t)}=
\frac{q_\phi(e\mid c_t)p_{\psi_e}(y_t\mid c_t,e)}{p_\Theta(y_t\mid c_t)}
\nabla_\Theta \log q_\phi(e\mid c_t)$


---

#### 2\. 第二项：expert 概率的梯度项

第二项是：

$\frac{q_\phi(e\mid c_t)\nabla_\Theta p_{\psi_e}(y_t\mid c_t,e)}{p_\Theta(y_t\mid c_t)}$

同理，乘除一个 $p_{\psi_e}(y_t\mid c_t,e)$：

$\frac{q_\phi(e\mid c_t)\nabla_\Theta p_{\psi_e}(y_t\mid c_t,e)}{p_\Theta(y_t\mid c_t)}=
\frac{q_\phi(e\mid c_t)p_{\psi_e}(y_t\mid c_t,e)}{p_\Theta(y_t\mid c_t)}
\frac{\nabla_\Theta p_{\psi_e}(y_t\mid c_t,e)}{p_{\psi_e}(y_t\mid c_t,e)}$

所以第二项可以写成：

$\frac{q_\phi(e\mid c_t)\nabla_\Theta p_{\psi_e}(y_t\mid c_t,e)}{p_\Theta(y_t\mid c_t)}=
\frac{q_\phi(e\mid c_t)p_{\psi_e}(y_t\mid c_t,e)}{p_\Theta(y_t\mid c_t)}
\nabla_\Theta \log p_{\psi_e}(y_t\mid c_t,e)$

---

#### 3\. 合并两项

因此：

$\nabla_\Theta \log p_\Theta(y_t\mid c_t)=
\sum_e
\frac{q_\phi(e\mid c_t)p_{\psi_e}(y_t\mid c_t,e)}{p_\Theta(y_t\mid c_t)}
\left[
\nabla_\Theta \log q_\phi(e\mid c_t)
+
\nabla_\Theta \log p_{\psi_e}(y_t\mid c_t,e)
\right]$


注意分母：


$p_\Theta(y_t\mid c_t)=\sum_j q_\phi(j\mid c_t)p_{\psi_j}(y_t\mid c_t,j)$

所以前面的系数可以写成：

$\frac{q_\phi(e\mid c_t)p_{\psi_e}(y_t\mid c_t,e)}{p_\Theta(y_t\mid c_t)}
=\frac{q_\phi(e\mid c_t)p_{\psi_e}(y_t\mid c_t,e)}{\sum_j q_\phi(j\mid c_t)p_{\psi_j}(y_t\mid c_t,j)}$

定义：

$\gamma_{t,e}=
\frac{q_\phi(e\mid c_t)p_{\psi_e}(y_t\mid c_t,e)}{\sum_j q_\phi(j\mid c_t)p_{\psi_j}(y_t\mid c_t,j)}$

那么根据贝叶斯公式，它就是：

$\gamma_{t,e}=p_\Theta(E=e\mid c_t,y_t)$

也就是说，$\gamma_{t,e}$ 表示看到真实 token $y_t$ 之后，expert $e$对当前 token 的后验责任。

因此最后得到：

${
\nabla_\Theta \log p_\Theta(y_t\mid c_t)=
\sum_e
\gamma_{t,e}
\left[
\nabla_\Theta \log q_\phi(e\mid c_t)
+
\nabla_\Theta \log p_{\psi_e}(y_t\mid c_t,e)
\right]
}$

当开启R3时，第一项$\nabla_\Theta \log q_\phi(e\mid c_t)$ 会牺牲一部分router自由选择带来的梯度信号。

---

### 分析 log probability 对 Router 参数的梯度

第五部分最后得到的是对整个 MoE 参数 $\Theta$ 的梯度，包括 router 参数和 expert 参数：

$\Theta=\{\phi,\psi_1,\psi_2,\dots,\psi_N\}$

因此，上式实际上同时包含两类梯度信号：

- 第一项 $\nabla_\Theta \log q_\phi(e\mid c_t)$ 对应 router 分支；

- 第二项 $\nabla_\Theta \log p_{\psi_e}(y_t\mid c_t,e)$ 对应 expert 分支。

此处我们只关心 router 参数 $\phi$，对其单独求导：



由于 expert 条件概率$p_{\psi_e}(y_t\mid c_t,e)$ 由第 $e$ 个 expert 的参数 $\psi_e$ 决定，在当前概率模型中它不直接依赖 router 参数 $\phi$，所以：

$\nabla_\phi \log p_{\psi_e}(y_t\mid c_t,e)=0$

因此，对 router 参数求梯度时，整体公式中的 expert 项会消失，只剩下 router 项：

${\nabla_\phi \log p_\Theta(y_t\mid c_t)=\sum_e \gamma_{t,e}\nabla_\phi \log q_\phi(e\mid c_t)}$

其中：

$\gamma_{t,e}=\frac{q_\phi(e\mid c_t)p_{\psi_e}(y_t\mid c_t,e)}{\sum_j q_\phi(j\mid c_t)p_{\psi_j}(y_t\mid c_t,j)}$

它表示看到真实 token $y_t$ 之后，expert $e$ 对当前 token 的后验责任。

所以 router 参数接收到的训练信号可以理解为：router 的 log probability 梯度，是所有 expert 的 router log\-prob 梯度的加权和；权重就是每个 expert 的后验责任 $\gamma_{t,e}$。如果某个 expert 在看到真实 token 后具有更大的后验责任，那么它对应的 router log\-prob 梯度权重也更大。也就是说，router 会被推动去提高这类 expert 在类似上下文中的选择概率。

如果训练目标是负对数似然：

$\mathcal{L}t=-\log p\Theta(y_t\mid c_t)$

那么 router 参数的梯度为：

${\nabla_\phi \mathcal{L}t=-\sum_e \gamma_{t,e}\nabla_\phi \log q_\phi(e\mid c_t)}$

这里需要注意，$\gamma_{t,e}$ 虽然由 $q_\phi$ 和 $p_{\psi_e}$ 定义，但在这个公式里它是由原始梯度整理得到的系数，不是额外引入的新 loss，也不需要在这个表达式中再额外对 $\gamma_{t,e}$ 求导。

---

### 展开 softmax Router：从 router\-only 梯度到 logits 梯度

已知 log probability 对 router 参数 $\phi$ 的梯度：

$\nabla_\phi \log p_\Theta(y_t\mid c_t)=\sum_e \gamma_{t,e}\nabla_\phi \log q_\phi(e\mid c_t)$

接下来继续假设 router 是一个 softmax router。也就是说，router 先根据当前 token 的 hidden state $h_t$ 计算 logits：

$z_t=W_rh_t+b_r$

其中，$z_t\in\mathbb{R}^N$，第 $j$ 个 expert 的 logit 记为 $z_{t,j}$。

然后通过 softmax 得到 expert 选择概率：

$q_{t,e}=q_\phi(e\mid c_t)=\frac{\exp(z_{t,e})}{\sum_{k=1}^{N}\exp(z_{t,k})}$

为了简化记号，下面把 $q_\phi(e\mid c_t)$ 写成 $q_{t,e}$。

#### 先求 $\log q_{t,e}$ 对 router logit $z_{t,j}$ 的梯度

由 softmax 定义：

$q_{t,e}=\frac{\exp(z_{t,e})}{\sum_k\exp(z_{t,k})}$

两边取对数：

$\log q_{t,e}=\log\frac{\exp(z_{t,e})}{\sum_k\exp(z_{t,k})}$

利用 $\log\frac{a}{b}=\log a-\log b$，得到：

$\log q_{t,e}=z_{t,e}-\log\sum_k\exp(z_{t,k})$

现在对 $z_{t,j}$ 求偏导。

第一项为：

$\frac{\partial z_{t,e}}{\partial z_{t,j}}=\mathbf{1}_{e=j}$

其中：$\mathbf{1}_{e=j}=1$ 当 $e=j$，否则为 $0$。

第二项为：

$\frac{\partial}{\partial z_{t,j}}\log\sum_k\exp(z_{t,k})=\frac{1}{\sum_k\exp(z_{t,k})}\frac{\partial}{\partial z_{t,j}}\sum_k\exp(z_{t,k})$

因为求和中只有第 $j$ 项依赖 $z_{t,j}$，所以：

$\frac{\partial}{\partial z_{t,j}}\sum_k\exp(z_{t,k})=\exp(z_{t,j})$

因此：

$\frac{\partial}{\partial z_{t,j}}\log\sum_k\exp(z_{t,k})=\frac{\exp(z_{t,j})}{\sum_k\exp(z_{t,k})}=q_{t,j}$

所以：

${\frac{\partial \log q_{t,e}}{\partial z_{t,j}}=\mathbf{1}_{e=j}-q_{t,j}}$

这就是 log\-softmax 对 logits 的标准求导结果。

---

#### 将 router\-only 梯度继续展开到 logits

第3\.3节中提到 router\-only 梯度是：

$\nabla_\phi \log p_\Theta(y_t\mid c_t)=\sum_e \gamma_{t,e}\nabla_\phi \log q_{t,e}$

根据链式法则，log probability 对 router logits $z_{t,j}$ 的梯度：

$\frac{\partial \log p_\Theta(y_t\mid c_t)}{\partial z_{t,j}}=\sum_e \gamma_{t,e}\frac{\partial \log q_{t,e}}{\partial z_{t,j}}$

代入刚才得到的 log\-softmax 梯度：

$\frac{\partial \log p_\Theta(y_t\mid c_t)}{\partial z_{t,j}}=\sum_e \gamma_{t,e}(\mathbf{1}_{e=j}-q_{t,j})$

$\frac{\partial \log p_\Theta(y_t\mid c_t)}{\partial z_{t,j}}=\sum_e \gamma_{t,e}\mathbf{1}_{e=j}-\sum_e \gamma_{t,e}q_{t,j}$

第一项中，只有 $e=j$ 时指示函数为 $1$，所以：

$\sum_e \gamma_{t,e}\mathbf{1}_{e=j}=\gamma_{t,j}$

第二项中，$q_{t,j}$ 与求和变量 $e$ 无关，可以提出求和号：

$\sum_e \gamma_{t,e}q_{t,j}=q_{t,j}\sum_e \gamma_{t,e}$

由于 $\gamma_{t,e}$ 是一个后验责任分布，所以：

$\sum_e \gamma_{t,e}=1$

因此：

$\sum_e \gamma_{t,e}q_{t,j}=q_{t,j}$

最终得到：

${\frac{\partial \log p_\Theta(y_t\mid c_t)}{\partial z_{t,j}}=\gamma_{t,j}-q_{t,j}}$

如果写成向量形式，则为：

${\frac{\partial \log p_\Theta(y_t\mid c_t)}{\partial z_t}=\gamma_t-q_t}$

其中：

$q_t=[q_{t,1},q_{t,2},\dots,q_{t,N}]$

$\gamma_t=[\gamma_{t,1},\gamma_{t,2},\dots,\gamma_{t,N}]$

---

#### 对负对数似然 loss 的 logits 梯度

训练语言模型时，单个 token 的 loss 是：

$\mathcal{L}t=-\log p\Theta(y_t\mid c_t)$

因此：

$\frac{\partial \mathcal{L}t}{\partial z{t,j}}=-\frac{\partial \log p_\Theta(y_t\mid c_t)}{\partial z_{t,j}}$

代入7\.2的结果得：

$\frac{\partial \mathcal{L}t}{\partial z{t,j}}=q_{t,j}-\gamma_{t,j}$

向量形式为：

${\frac{\partial \mathcal{L}_t}{\partial z_t}=q_t-\gamma_t}$

---

#### 3\.4\.4  从 logits 梯度继续推到 Router 权重矩阵 $W_r$ 的梯度

现在已经得到了 loss 对 router logits 的梯度：

$\frac{\partial \mathcal{L}_t}{\partial z_t}=q_t-\gamma_t$

接下来要继续把梯度传到 router 的参数 $W_r$ 和 $b_r$。

router 的线性层为：

$z_t=W_rh_t+b_r$

其中：

- $h_t\in\mathbb{R}^d$ 是当前 token 的 hidden state；

- $W_r\in\mathbb{R}^{N\times d}$ 是 router 权重矩阵；

- $b_r\in\mathbb{R}^N$ 是 router 偏置；

- $z_t\in\mathbb{R}^N$ 是 router logits。

---

用元素形式推导 $W_r$ 的梯度

第 $j$ 个 expert 的 logit 为：

$z_{t,j}=W_{r,j:}h_t+b_{r,j}$

展开成分量形式：

$z_{t,j}=\sum_{m=1}^{d}W_{r,jm}h_{t,m}+b_{r,j}$

现在要求 loss 对 $W_{r,jm}$ 的梯度。根据链式法则：

$\frac{\partial \mathcal{L}t}{\partial W{r,jm}}=\sum_l \frac{\partial \mathcal{L}t}{\partial z{t,l}}\frac{\partial z_{t,l}}{\partial W_{r,jm}}$

由于参数 $W_{r,jm}$ 只影响第 $j$ 个 logit $z_{t,j}$，不影响其他 $z_{t,l}$，所以只有 $l=j$ 这一项保留：

$\frac{\partial \mathcal{L}t}{\partial W{r,jm}}=\frac{\partial \mathcal{L}t}{\partial z{t,j}}\frac{\partial z_{t,j}}{\partial W_{r,jm}}$

而：

$\frac{\partial z_{t,j}}{\partial W_{r,jm}}=h_{t,m}$

因此：

$\frac{\partial \mathcal{L}t}{\partial W{r,jm}}=\frac{\partial \mathcal{L}t}{\partial z{t,j}}h_{t,m}$

所以：

${\frac{\partial \mathcal{L}t}{\partial W{r,jm}}=(q_{t,j}-\gamma_{t,j})h_{t,m}}$

写成矩阵形式

将上面的元素形式写成矩阵形式：

${\frac{\partial \mathcal{L}_t}{\partial W_r}=(q_t-\gamma_t)h_t^\top}$

这里：

- $q_t-\gamma_t\in\mathbb{R}^{N}$；

- $h_t^\top\in\mathbb{R}^{1\times d}$；

- 所以 $(q_t-\gamma_t)h_t^\top\in\mathbb{R}^{N\times d}$，形状正好和 $W_r$ 一致。

对偏置 $b_r$:

${\frac{\partial \mathcal{L}_t}{\partial b_r}=q_t-\gamma_t}$

如果有多个 token，则梯度在所有 token 上求和：

${\frac{\partial \mathcal{L}}{\partial W_r}=\sum_t (q_t-\gamma_t)h_t^\top}$

${\frac{\partial \mathcal{L}}{\partial b_r}=\sum_t (q_t-\gamma_t)}$

## 真实的MOE层混合概率模型

前述推导有助于理解 router 的概率解释，但我们把 expert 近似成直接输出 token probability。实际 Transformer MoE 中，expert 通常是 FFN，输出的是 hidden representation。因此，后续分析 R3 时，我们转向真实 MoE 层的 hidden\-vector mixing 建模。

设 Transformer 中的 MoE 层集合为：

$\mathcal{M}\subseteq \{1,2,\dots,L\}$

其中，$l\in\mathcal{M}$ 表示第 $l$ 层是 MoE 层。

第 $l$ 层、第 $t$ 个 token 进入 MoE 层前的 hidden state 记为：$h_t^{(l)}$。

第 $l$ 层 router 计算 logits：

$z_t^{(l)}=W_r^{(l)}h_t^{(l)}+b_r^{(l)}$

经过 softmax 得到 router 概率：

$q_t^{(l)}=\operatorname{softmax}(z_t^{(l)})$

当前 router 根据 $q_t^{(l)}$ 选择 Top\-k expert 集合：

$S_t^{cur,(l)}=\operatorname{TopK}(q_t^{(l)})$

第 $e$ 个 expert 的输出记为：

$u_{t,e}^{(l)}=E_e^{(l)}(h_t^{(l)})$

其中，$u_{t,e}^{(l)}$ 是 hidden representation，$E_e^{(l)}$ 表示：Transformer 第 $l$ 层 MoE 中，第$e$ 个专家对应的模型函数。

如果 Top\-k 后重新归一化 gate 权重，则对 $e\in S_t^{cur,(l)}$：

$\tilde q_{t,e}^{cur,(l)}=\frac{\exp z_{t,e}^{(l)}}{\sum_{j\in S_t^{cur,(l)}}\exp z_{t,j}^{(l)}}$

于是 No\-R3 情况下，第 $l$ 层 MoE 的输出为：



$o_t^{NoR3,(l)}=\sum_{e\in S_t^{cur,(l)}}\tilde q_{t,e}^{cur,(l)}E_e^{(l)}(h_t^{(l)})$



这是真实 Transformer MoE 层中常见的建模形式。

---

## No\-R3 与 R3 的前向路径和 Router 梯度差异

在 No\-R3 情况下，训练阶段的 expert 集合由当前 router 自己决定：

$S_t^{cur,(l)}=\operatorname{TopK}(q_t^{(l)})$

因此 No\-R3 的 MoE forward 是：

$o_t^{NoR3,(l)}=\sum_{e\in S_t^{cur,(l)}}\tilde q_{t,e}^{cur,(l)}E_e^{(l)}(h_t^{(l)})$

也就是说，当前 router 的 expert dispatch decision 直接决定当前训练 step 的 forward path。

而在 R3 情况下，训练阶段不再使用当前 router 自己选出的 expert 集合 $S_t^{cur,(l)}$，而是使用 rollout 阶段 replay 下来的 expert ID：$S_t^{R3,(l)}$

R3 只 replay expert ID，不 replay gate weight。因此，R3 固定的是 expert dispatch set，但 gate weight 仍然由当前 router logits 计算。对 $e\in S_t^{R3,(l)}$，有：

$\tilde q_{t,e}^{R3,(l)}=\frac{\exp z_{t,e}^{(l)}}{\sum_{j\in S_t^{R3,(l)}}\exp z_{t,j}^{(l)}}$

所以 R3 下的 MoE forward 是：

$o_t^{R3,(l)}=\sum_{e\in S_t^{R3,(l)}}\tilde q_{t,e}^{R3,(l)}E_e^{(l)}(h_t^{(l)})$

这说明 R3 并没有完全切断 router 的梯度。因为 $\tilde q_{t,e}^{R3,(l)}$ 仍然依赖当前 router logits $z_t^{(l)}$。

但是 R3 改变了一个关键点：训练 forward 使用的 expert 集合不再由当前 router 的 $S_t^{cur,(l)}$ 决定，而是由 replay 的 $S_t^{R3,(l)}$ 决定。

为了更清晰地推导梯度，设第 $l$ 层 MoE 输出处的上游梯度为：

$\delta_t^{(l)}=\frac{\partial \mathcal L}{\partial o_t^{(l)}}$

这里 $\delta_t^{(l)}$ 已经包含了后续 Transformer 层、LayerNorm、LM head 和 loss 的影响。

定义：

$a_{t,e}^{(l)}=(\delta_t^{(l)})^\top E_e^{(l)}(h_t^{(l)})=\frac{\partial \mathcal{L}_t^{(l)}}{q_{t,e}^{(l)}}$

它表示：如果增大 expert $e$ 的 gate 权重，loss 会如何变化。

在 No\-R3 中，对 $j\in S_t^{cur,(l)}$，router logit 的梯度为：

$\frac{\partial \mathcal L^{NoR3}}{\partial z_{t,j}^{(l)}}=\tilde q_{t,j}^{cur,(l)}\left(a_{t,j}^{(l)}-\sum_{e\in S_t^{cur,(l)}}\tilde q_{t,e}^{cur,(l)}a_{t,e}^{(l)}\right)$

对于未被当前 router 选中的 expert：

$\frac{\partial \mathcal L^{NoR3}}{\partial z_{t,j}^{(l)}}=0,\quad j\notin S_t^{cur,(l)}$

而在 R3 中，只需要把当前 router 的 expert 集合 $S_t^{cur,(l)}$ 替换为 replay 集合 $S_t^{R3,(l)}$。于是，对 $j\in S_t^{R3,(l)}$，有：

$\frac{\partial \mathcal L^{R3}}{\partial z_{t,j}^{(l)}}=\tilde q_{t,j}^{R3,(l)}\left(a_{t,j}^{(l)}-\sum_{e\in S_t^{R3,(l)}}\tilde q_{t,e}^{R3,(l)}a_{t,e}^{(l)}\right)$

对于未在 replay expert 集合中的 expert：

$\frac{\partial \mathcal L^{R3}}{\partial z_{t,j}^{(l)}}=0,\quad j\notin S_t^{R3,(l)}$

进一步，令：

$r_t^{(l)}=\frac{\partial \mathcal L}{\partial z_t^{(l)}}$

则 router 权重矩阵的梯度为：

$\frac{\partial \mathcal L}{\partial W_r^{(l)}}=\sum_t r_t^{(l)}(h_t^{(l)})^\top$

偏置项梯度为：

$\frac{\partial \mathcal L}{\partial b_r^{(l)}}=\sum_t r_t^{(l)}$

如果整个模型有多个 MoE 层，那么整体 router 参数为：$\phi=\{\phi^{(l)}\}_{l\in\mathcal M}$，因此整体 router 梯度可以写为：

$\nabla_\phi \mathcal L=\sum_{l\in\mathcal M}\nabla_{\phi^{(l)}}\mathcal L$

因此，更准确的结论是：R3保留了replay expert集合内部的gate\-weight梯度，但削弱了current router dispatch decision与training forward path之间的耦合。

进一步说，R3 的影响不是单层的，而是发生在每一个 MoE 层：$l\in\mathcal M$

因此，R3 replay 的对象可以看成一条多层 route trajectory：

$\mathcal R^{R3}=\{S_t^{R3,(l)}\}_{l\in\mathcal M,t}$

而当前 router 自己会产生的 route trajectory 是：

$\mathcal R^{cur}=\{S_t^{cur,(l)}\}_{l\in\mathcal M,t}$

两者之间的差异可以用 route overlap 来刻画：

$\operatorname{Overlap}=\frac{1}{|\mathcal M|T}\sum_{l\in\mathcal M}\sum_t\frac{|S_t^{cur,(l)}\cap S_t^{R3,(l)}|}{k}$

因此，思考后续课题是否可以围绕此展开：

如何在保持R3 replay dispatch consistency的同时，补偿current router dispatch decision 与training forward path解耦带来的梯度损失信号？



