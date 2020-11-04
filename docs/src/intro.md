# Introduction

An optimization problem is the problem of finding the best solution from all feasible solutions. The standard form of an optimization problem is 
```math
\begin{aligned}
&{\underset {x}{\operatorname {minimize} }}&&f(x)\\&\operatorname {subject\;to} &&g_{i}(x)\leq 0,\quad i=1,\dots ,m\\&&&h_{j}(x)=0,\quad j=1,\dots ,p
\end{aligned}
```

Note that finding solution to most of the optimization problems is computationally intractable. Here we consider a subset of those problems called [convex optimization problems](https://en.wikipedia.org/wiki/Convex_optimization), which admit polynomial time solutions. The standard form of a convex optimization problem is 
```math
\begin{aligned}
&{\underset {x}{\operatorname {minimize} }}&&f(x)\\&\operatorname {subject\;to} &&g_{i}(x)\leq 0,\quad i=1,\dots ,m\\&&&A x = b
\end{aligned}
```
where $f$ and $g_{i}$ are [convex functions](https://en.wikipedia.org/wiki/Convex_function).

## Parameterized  problems
In practice, convex optimization problems include parameters, apart from the decision variables, which determines the structure of the problem itself i.e. the objective function and constraints. Hence they affect the solution too. A general form of a parameterized convex optimization problem is 
```math
\begin{aligned}
&{\underset {x}{\operatorname {minimize} }}&&f(x; \theta)\\&\operatorname {subject\;to} &&g_{i}(x; \theta)\leq 0,\quad i=1,\dots ,m\\&&&A(\theta) x = b(\theta)
\end{aligned}
```
where $\theta$ is the parameter. In different fields, these parameters go by different names:

1. Hyperparameters in machine learning
2. Risk aversion or other backtesing parameters in financial modelling
3. Parameterized systems in control theory

## What do we mean by differentiating a parameterized optimization program? Why do we need it?
Often, parameters are chosen and tuned by hand - an iterative process - and the structure of the problem is crafted manually. But it is possible to do an *automatic gradient based tuning* of parameters.

Consider solution of the parametrized optimization problem, $x(\theta)$,

```math
\begin{split}
\begin{array} {lll}
x^*(\theta)=& {\underset {x}{\operatorname {argmin} }}& f(x; \theta)\\
            &  \operatorname {subject\;to} & g_{i}(x; \theta)\leq 0,\quad i=1,\dots ,m\\
            &                              & A(\theta) x = b(\theta)
\end{array}
\end{split}
```
which is the input of $l(x^*(\theta))$, a loss function. Our goal is to choose the best parameter $\theta$ so that $l$ is optimized. Here, $l(x^*(\theta))$ is the objective function and $\theta$ is the decision variable. In order to apply a gradient-based strategy to this problem, we need to differentiate $l$ with respect to $\theta$.
```math
\frac{\partial l(x^*(\theta))}{\partial \theta} = \frac{\partial l(x^*(\theta))}{\partial x^*(\theta)}  \frac{\partial x^*(\theta)}{\partial \theta}
```

By [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem#Definitions), this translates to differentiating the program data, i.e. functions $f$, $g_i(x)$ and matrices $A$, $b$, with respect to $\theta$.

This is can be achieved in two steps or passes:

1. Forward pass - Given an initial value of $\theta$, solves the optimization problem to find $x^*(\theta)$
2. Backward pass - Given $x^*$, differentiate and find $\frac{\partial x^*(\theta)}{\partial \theta}$