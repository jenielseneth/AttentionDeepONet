## DeepONet
The DeepONet formalisation is:

$$G(u)(y) \approx \sum_{k=1}^p b_k(u(x_1),u(x_2), \dots, u(x_m))t_k(y) \in \R$$

For DeepONet to be a parallel of Attention Mechanism we work backwards. 
Let's first note the dimensionality of each of the elements.

$$b_k \in \R, \ t_k \in \R, x, y \in \R^d$$

Assume that:
$$
\text{branch net} = (\psi(K)^T V) \in \R^{R \times d}\\
\text{trunk net} = \phi(Q)  \in \R^{N \times R}
$$


We focus on a single input y, $N = 1$. The trunk net produces a $\R^p$ output, meaning that the normalised Query Matrix should be of size $\R^{1 \times p}$. This extends to multiple inputs, where the normalised query matrix is then of size $\R^{N \times p}$.

Similarly, the branch net produces a $\R^p$ output. We want the final attention output to be a single scalar, which means that the KTV term should be of size $\R^{p \times 1}$. K should be of size $\R^{m \times p}$ and V should be of size $\R^{m \times 1}$.

The official definition of a single branch coefficient $b_k$ and trunk coefficient $t_k$ is:

$$
b_k(u(x_1), u(x_2), \dots, u(x_m))
=
\sum_{i=1}^{n}
c_i^k
\sigma
(
\sum_{j=1}^m
\xi^k_{ij}
u(x_j) 
+
\theta^k_i
)
\\

t_k(y)
=
\sigma
(
w_k
\cdot
y
+
\zeta_k
)
$$

We need to rewrite this in terms of a kernel matrix $\phi$ and $\psi$ respectively.

For our trunk net, $\phi$ and Q are derived easily:

$$
Q = Y = 
\begin{bmatrix}
y_1^T \\
y_2^T \\
\vdots \\
y_N^T 
\end{bmatrix}
\in \R^{N \times d}

\\

W = \begin{bmatrix}
w_1^T \\
w_2^T \\
\vdots \\
w_p^T 
\end{bmatrix}
\in \R^{p \times d}

\\
z = \begin{bmatrix}
\zeta_1, \zeta_2, \dots, \zeta_p
\end{bmatrix}, 
Z = \begin{bmatrix}
z \\
z \\
\vdots \\
z
\end{bmatrix} \in \R^{N \times p}
\\
\phi(X) = \sigma(XW^T + Z)
$$

Next we tackle writing $\psi$ as the branch net function. We first rewrite the branch net term as a matrix product:

$$
b_k = \sum_{i=1}^{n}
c_i^k
\sigma
(
\sum_{j=1}^m
\xi^k_{ij}
u(x_j) 
+
\theta^k_i
)
\\
C = 
\begin{bmatrix}
c_1^1 & c_2^1 & \dots & c_n^1 \\
c_1^2 & c_2^2 & \dots & c_n^2 \\
\vdots \\

c_1^p & c_2^p & \dots & c_n^p
\end{bmatrix} \in \R^{p \times n}
\\
\Zeta \in \R^{p \times n \times m} 
\\
\theta \in \R^{p \times n}
\\
U \in \R^{m}
\\

$$

$$
\sum_{j=1}^m
\xi^k_{ij}
u(x_j) 
+
\theta^k_i
= (\Zeta_k U + \theta_k)_i,  \Zeta_k U + \theta_k \in \R^{n}

\\
ZU + \theta = 
\begin{bmatrix}
\sum_{j=1}^m \xi^1_{1j} u(x_j) + \theta^1_1 & \sum_{j=1}^m \xi^1_{2j} u(x_j) + \theta^1_2 & \dots \\
\sum_{j=1}^m \xi^2_{1j} u(x_j) + \theta^2_1 & \sum_{j=1}^m \xi^2_{2j} u(x_j) + \theta^2_2 & \dots
\end{bmatrix}
 \in \R^{p \times n} 
\\

\sum_{i=1}^{n}
c_i^k
\sigma
(
\sum_{j=1}^m
\xi^k_{ij}
u(x_j) 
+
\theta^k_i
)
= (\sigma(\Zeta U + \theta) C^T)_{kk} \text{ (get diagonal)}  
\\ 
b = diag(\sigma(\Zeta U + \theta) C^T) \in \R^p
$$

Because each coefficient with a k-index is also dependent on another index, it isn't possible to write any matrix as the Value matrix and Key matrix: K should be of size $\R^{m \times p}$ and V should be of size $\R^{m \times 1}$.

However, if we rewrite the matrices as the following:

$$
b_k = \sum_{i=1}^{n}
c_i^k
\sigma
(
\sum_{j=1}^m
\xi^k_{ij}
u(x_j) 
+
\theta^k_i
)
\\
C = 
\begin{bmatrix}
c_1^1 \\
\vdots \\
c_n^1 \\
c_1^2 \\
\vdots \\
c_1^p \\
\vdots \\
c_n^p \\
\end{bmatrix} \in \R^{p* n}
\\
\sum_{j=1}^m
\xi^k_{ij}
u(x_j) 
+
\theta^k_i
= (\Zeta_k U + \theta_k)_i,  \Zeta_k U + \theta_k \in \R^{n}

\\
A = 
\begin{bmatrix}
\sum_{j=1}^m \xi^1_{1j} u(x_j) + \theta^1_1 & \sum_{j=1}^m \xi^1_{2j} u(x_j) + \theta^1_2 & \dots & \sum_{j=1}^m \xi^1_{nj} u(x_j) + \theta^1_n & 0 & \dots \\
0 & \dots & & 0 & \sum_{j=1}^m \xi^2_{1j} u(x_j) + \theta^2_1  & \dots & \sum_{j=1}^m \xi^2_{nj} u(x_j) + \theta^2_n & \dots \\
\vdots \\
0 & 0 & \dots & & & & & \dots & \sum_{j=1}^m \xi^p_{1j} u(x_j) + \theta^p_1  & \dots & \sum_{j=1}^m \xi^p_{nj} u(x_j) + \theta^p_n
\end{bmatrix}
 \in \R^{p \times p* n} 
\\

b = \sigma(A) C \in \R^p
$$

This gives us a proper analog to the key and value matrices:
$$
K=A^T, V=C, \psi = \sigma
$$

If we write $K = U$, we could potentially rewrite $\psi$ as:
$$
\psi(X) = \sigma(
\begin{bmatrix}
(\Zeta_1 X + \theta_1)^T & 0 & \dots & & 0\\
0 & (\Zeta_2 X + \theta_2)^T & 0 & \dots & \vdots\\
\vdots & & \ddots & \\
0 & \dots & & & (\Zeta_p X + \theta_p)^T
\end{bmatrix}
)
$$

If our goal is to rewrite $\psi(K)^TV$ as a $\R^{p \times p}$ matrix so that each element ij of the final attention matrix is $t_j(y_i)b_j$, we can rewrite C as:

$$
C = 
\begin{bmatrix}
c_1^1 & 0 & \dots\\
\vdots \\
 c_n^1 & 0  & \dots\\
0 & c_1^2 & 0 & \dots  \\
\vdots &  \\
0 & c_1^p & 0 & \dots\\
\vdots & \vdots & \vdots & \\
0 & \dots & & c_1^p \\
& &  \dots & \\
0 & \dots & & c_n^p \\
\end{bmatrix} 
= 
\begin{bmatrix}
c^1 & 0 & \dots & 0\\
0 & c^2 & \dots & 0 \\
& & \ddots & \vdots & \\
0 & \dots & & c^p \\
\end{bmatrix} 
\in \R^{p*n \times p}

$$

## Mixing Tensor

We embed a mixing tensor $A \in \R^{p \times p \times p \times p}$ in between $\phi(Q) \in \R^{ N \times p}$ and $\psi(K)^TV \in \R^{p\times p}$: 

$$
\text{attention(Q, K)} = \phi(Q)A(\psi(K)^TV)
$$

In Einstein notation, $A(\psi(K)^TV)$ would be written as $\text{einsum}(ijkl, kl -> ij)$. In mathematical formulation, this is then:
$$
C_{ij} = \sum_{k=1}^p \sum_{l=1}^p A_{ijkl} (\psi(K)^TV)_{kl}
$$

 Because $\psi(K)^TV$ is a diagonal, whenever $k \neq l$ we have $K_{kl} = 0$. This leads to a contraction of our formulation:

$$
C_{ij} = \sum_{k=1}^p A_{ijkk} (\psi(K)^TV)_{kk}
$$
This means that only the diagonals of A in each sub-tensor at position $i,j$ are actually used, meaning we don't need to store a 4D tensor. Instead we can think of A as a 3D tensor: we denote the diagonal of \psi(K)^TV as v. We can effectively write: $C_i = A_i v$. 


## Multi-layered DeepONet from an Attention perspective

If $\psi(K)^TV$ gives us an $p \times p$ matrix, our final output $\phi(Q) \psi (K)^T V$ gives us a $N \times p$ matrix. If we interpret this as a single Attention layer, this can be thought of as our updated $Q$ matrix. We can then pass this updated matrix through another Attention layer.  
 
##

We can even go one step further and rethink DeepONets as Chebyshev expansions. We reinterpret the trunk coefficients as the evaluations of our point $y$ of set of basis functions $t_1(y), t_2(y), \dots$. Let's assume that we want to evaluate at a point $y = \theta, \theta \in [-\pi, \pi]$. If we want to evaluate this point at a Chebyshev polynomial $T_n(y) = T_n(\cos(\theta)) = \cos(n\theta) = $.

Let's look at the trunk net definition again:

$$
t_k(y)
=
\sigma
(
w_k
\cdot
y
+
\zeta_k
)
$$