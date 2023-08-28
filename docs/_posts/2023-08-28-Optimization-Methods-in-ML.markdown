---
title: "Optimization Methods in ML"
categories: ML
---

# Gradient Descent

For a loss function $Q(w)$,

$$
{\displaystyle w:=w-\eta \nabla Q(w)=w-{\frac {\eta }{n}}\sum _{i=1}^{n}\nabla Q_{i}(w)}
$$

Here, $Q_i(w)$ is associated with the $i$-the observation in the data set.

# Stochastic Gradient Descent

Instead of considering the entire data set $\frac1n \sum_{i=1}^n \nabla Q_i(w)$, update $w$

$$
{\displaystyle w:=w-\eta \nabla Q_i(w)}
$$

1. Choose an initial vector of parameters $w$ and learning rate $\eta$.
2. Repeat until an approximate minimum is obtained:
   a. Randomly _shuffle samples_ in the training set.
   b. For $i = 1, 2, ..., n$: - Update the parameter vector: $w := w - η ∇Qᵢ(w)$.

# Momentum

$$
\begin{align*}
&\begin{cases}
&\Delta w:=\alpha \Delta w-\eta \nabla Q_{i} \\
&w:=w+\Delta w
\end{cases} \\
&\Rightarrow w:=w + \alpha \Delta w -\eta \nabla Q_{i}(w)
\end{align*}
$$

# Adagrad

**AdaGrad** (for **adaptive gradient algorithm**)

- increases the learning rate for sparse parameters;
- decreases the learning rate for ones that are less sparse.

This strategy often improves convergence performance over standard stochastic gradient descent in settings where data is sparse and sparse parameters are more informative.

It still has a **base learning rate** $\eta$, but this is multiplied by the elements of a vector $\{G_{j, j}\}$ which is the diagonal of the outer product matrix

$$
G = \sum_{\tau = 1}^t g_\tau g_\tau^\intercal
$$

where $g_\tau = \nabla Q_i(w)$, the gradient, at iteration $\tau$. The diagonal is given by

$$
G_{j, j} = \sum_{\tau = 1}^t g_{\tau, j}^2.
$$

This vector essentially stores a _historical sum of gradient squares by dimension_ and is updated after every iteration. The formula for an update is now

$$
w:=w - \eta \operatorname{diag}(G)^{-\frac12} \odot g
$$

or, written as per-parameter updates,

$$
w_j := w_j - \frac{\eta}{\sqrt{G_{j,j}}}g_j = w_j - \frac{\eta}{\sqrt{\sum_{\tau=1}^t g_\tau^t}}g_j.
$$

Each $\{G_{(i, i)}\}$ gives rise to a scaling factor for the learning rate that applies to a single parameter $w_i$. Since the denominator in this factor, $\sqrt{G_i} = \sqrt{\sum_{\tau=1}^t g_\tau^t}$, is the $\ell_2$ norm of previous derivatives, extreme parameter updates get dampened, while parameters that get few or small updates receive higher learning rates.

While designed for convex problems, AdaGrad has been successfully applied to non-convex optimization.

# RMSProp

In **RMSProp** (**for Root Mean Square Propagation**), the learning rate is, like in Adagrad, adapted for each of the parameters. The idea is to divide the learning rate for weight by a running average of the magnitudes of recent gradients for that weight.

The running average is first calculated in terms of means sqaure,

$$
v(w, t) := \gamma v(w, t-1) + (1-\gamma)(\nabla Q_i(w))^2
$$

where, $\gamma$ is the forgetting factor. The concept of storing the historical gradient as the sum of squares is borrowed from Adagrad, but \***\*\*\*\*\***forgetting\***\*\*\*\*\*** is introduced to solve Adagrad’s diminishing learning rates in non-convex problems by gradually decreasing the influence of old data.

$$
w:=w-\frac{\eta}{\sqrt{v(w, t)}} \nabla Q_i(w).
$$

# Adam

[](https://arxiv.org/pdf/1412.6980.pdf)

**Adam** (short for **Adaptive Moment Estimation**)

- RMSProp + momentum method
- Running averages with exponential forgetting of both the gradients and the second moments of the gradients are used.

Given parameters $w^{(t)}$ and a loss function $L^{(t)}$, where $t$ indexes the current training iteration (indexed at $0$), Adam’s parameter update is given by:

$$
\begin{align*}
&\text{momentum} & m_{w}^{(t+1)} \leftarrow \beta_{1}m_{w}^{(t)} + (1-\beta_{1})\nabla_{w}L^{(t)} \\
&\text{RMSProp} & v_{w}^{(t+1)} \leftarrow \beta_{2}v_{w}^{(t)} + (1-\beta_{2})(\nabla_{w}L^{(t)})^{2} \\
&\text{bias-correction} & \hat{m}_{w} = \frac{m_{w}^{(t+1)}}{1-\beta_{1}^{t}} \\
&\text{bias-correction} & \hat{v}_{w} = \frac{v_{w}^{(t+1)}}{1-\beta_{2}^{t}} \\
&\text{update } w & w^{(t+1)} \leftarrow w^{(t)} - \eta \frac{\hat{m}_{w}}{\sqrt{\hat{v}_{w}} + \epsilon}
\end{align*}
$$

where $\epsilon$ is a smaller scale (e.g. $10^{-8}$) used to prevent division by $0$, and $\beta_1$ (e.g. $0.9$) and $\beta_2$ (e.g. 0.999) are the forgetting factors for gradients and second moments of gradients, respectively. Squaring and sqaure-rooting is done element-wise.
