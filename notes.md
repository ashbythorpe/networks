---
title: Notes on Neural Networks
author: Ashby Thorpe
---

Let $C$ be our cost function, $x$ be our input, and $y$ be our output.
We want to find $\nabla_xC$, the gradient vector of $C$ with respect to $x$,
defined by:

$
\displaystyle
\nabla_xC=\begin{pmatrix}
\frac{\partial C}{\partial x_1}\\
\frac{\partial C}{\partial x_2}\\
\vdots\\
\frac{\partial C}{\partial x_n}
\end{pmatrix}
$

Now consider $\displaystyle \frac {dC}{dx}$, the total derivative. Since $C$ is
a scalar and $x$ is a vector, this must be a linear map from $\mathbb{R}^n$ to
$\mathbb{R}$ (a covector). We can deduce that
$\nabla_xC=\displaystyle (\frac {dC}{dx})^T$.

By the chain rule, we can work out $\displaystyle \frac {dC}{dx}$ to be:

${\displaystyle {\frac {dC}{da^{L}}}\cdot {\frac {da^{L}}{dz^{L}}}\cdot {\frac {dz^{L}}{da^{L-1}}}\cdot {\frac {da^{L-1}}{dz^{L-1}}}\cdot {\frac {dz^{L-1}}{da^{L-2}}}\cdot \ldots \cdot {\frac {da^{1}}{dz^{1}}}\cdot {\frac {\partial z^{1}}{\partial x}},}$

Where $a^l$ is the activation of layer $l$ and $z^l$ is the weighted input to
layer $l$ ($z^l$ is the input to the activation function).

${\displaystyle \frac {dC}{da^{L}}}$ is the derivative of the cost function with respect to
the activation of the last layer. As before, this is a covector.

${\displaystyle \frac {da^{L}}{dz^{L}}}$ is the derivative of the activation function (e.g.
the sigmoid function). Since this is applied to each element of a vector
individually, this is a diagonal matrix instead of a scalar. It is
sometimes represented as the Hadamard product.

${\displaystyle \frac {dz^{L}}{da^{L-1}}}$ is the derivative of the activation
of a layer with respect to its input. This is the derivative of the matrix
multiplication $W^L$, and since matrix multiplication is linear
(${\displaystyle (Wx)'=W}$), this is just the matrix $W^L$.

Then:

${\displaystyle \frac {dC}{dx} = {\frac {dC}{da^{L}}}\circ (f^{L})'\cdot W^{L}\circ (f^{L-1})'\cdot W^{L-1}\circ \cdots \circ (f^{1})'\cdot W^{1}.}$

And so the gradient is the transpose:

${\displaystyle \nabla _{x}C=(W^{1})^{T}\cdot (f^{1})'\circ \ldots \circ (W^{L-1})^{T}\cdot (f^{L-1})'\circ (W^{L})^{T}\cdot (f^{L})'\circ \nabla _{a^{L}}C.}$

Note that we can consider our input vector $x$ to be an arbitrary layer in the
network, and so we can use the same formula to find the gradient of the cost
function with respect to any layer in the network:

${\displaystyle \delta ^{l}:=(f^{l})'\circ (W^{l+1})^{T}\cdot (f^{l+1})'\circ \cdots \circ (W^{L-1})^{T}\cdot (f^{L-1})'\circ (W^{L})^{T}\cdot (f^{L})'\circ \nabla _{a^{L}}C.}$

Then we can calculate the gradient of the cost function with respect to the
weights of the network:

${\displaystyle \nabla _{W^{l}}C=\delta ^{l}(a^{l-1})^{T}.}$

Intuitively, if we denote this matrix $G$, then:

${\displaystyle g_{ij} = \delta^l_i a^{l-1}_j}$

This make sense since $w_{ij}$ is the weight of the $j$th neuron in layer $l-1$
to the $i$th neuron in layer $l$. If $W^l_{ij}$ changes by a small amount, then
the activation of the $i$th neuron in layer $l$ will change by $a^{l-1}_j$ times
that amount.

$\delta^l$ can be calculated recursively:

${\displaystyle \delta ^{l-1}:=(f^{l-1})'\circ (W^{l})^{T}\cdot \delta ^{l}.}$

By this process, the gradient of all the weights in the network can be
calculated.

## Bias

Note that the bias term can be considered as a weight with a constant input
of 1. By similar logic as before, the gradient vector is just $\delta^l$,
since if we change a component of the bias vector, this will directly affect
the corresponding term in $z^l$.

## Differentiating activation functions

If we use the logistic/sigmoid function as our activation function:

${\displaystyle \varphi (z)={\frac {1}{1+e^{-z}}}}$

Then:

${\displaystyle {\frac {d\varphi }{dz}}=\varphi (z)(1-\varphi (z))}$

## Differentiating cost functions

If our cost function is:

${\displaystyle C(a^{L},y)={\frac {1}{2}}\|a^{L}-y\|^{2}}$

Then:

${\displaystyle \frac {\partial C}{\partial a^{L}_i}=a^{L}_i-y}$

And so:

${\displaystyle \nabla _{a^{L}}C=a^{L}-y}$
