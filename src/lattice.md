---
title: Lattice Field Theory
papersize: A4
---

# Motivation

These lecture notes give an introduction to lattice field theory, a
powerful framework for solving quantum field theries from first
principles.
We approach the topic mainly from the point of view of QCD, going
through the building blocks necessary to simulate a model with fermions
with a gauge interaction.

# Learning Objectives

The course has two main objectives: to learn enough about lattice methods to put the into practical use and to become familiar with common methods used in studies of Standard Model physics.

After succesfully completing the course, the student

-   can implement a Montecarlo simulation of a quantum field theory in
    discrete space

-   can analyse the output of the calculation and describe it's
    connection to the theory

-   recognizes the most common discrete representations of field
    theories and is able to find information on them

-   recognizes observables in the discrete representations and is able
    to find information on them

-   can apply perturbation theory in a discrete space


# Spin models

We start with spin models as a simple example of a lattice model and to
get started with programming exercises.
This lesson introduces fundamental lattice simulation methods as well as important concepts from thermodynamics.

Spin models describe, in a rather simplified way, how a ferromagnet works.
We think of a clump of iron as a bunch of "atoms" sitting
unmoving on a structured lattice.

\[Image\]

Each atom has a spin, which creates a small magnetic field. When the
spins point to different random directions (as above), they cancel each
other and the iron is not magnetized. But if for some reason they point
to the same direction, the small magnetic fields add up into a
macroscopic value.

Why would this happen?

A spin pointing against a magnetic field has a slightly higher energy
than a spin pointing to agaist it. So atoms close to each other would
prefer to align. At zero temperature, they would eventually all point to
the same direction and create a magnet. At higher temperatures, thermal
fluctuations will occationally cause a spin to flip.

Let's only consider the spins closes to each other (nearest neighbours.)
The energy of two spins is

$$E=-j\boldsymbol{s}_{1}\cdot\boldsymbol{s}_{2}$$

and the energy of the whole system is

$$E=-J\sum_{<ij>}\boldsymbol{s}_{i}\boldsymbol{s}_{j}.$$ The sum over
$<ij>$ here counts pairs of neighbours. If the spins are coupled to a
magnetic field $\boldsymbol{H}$, with the magnetic momentum $\gamma$,
the energy is

$$E=-J\sum_{<ij>}\boldsymbol{s}_{i}\boldsymbol{s}_{j}-\gamma\boldsymbol{H}\cdot\sum_{i}\boldsymbol{s}_{i}.$$

At small temperatures the energy will tend to a minimum and the spins
will align. If there is a magnetic field, they will align to the
direction of the field. Otherwise they will pick a random direction.

At a non-zero temperature $T$ the *configurations* of spins will follow
the Boltzmann distribution,

$$\begin{align}
Z & =\int[\prod_{i}ds_{i}]e^{-\frac{1}{kT}E(s)}\\
 & =\int[\prod_{i}ds_{i}]e^{\frac{J}{kT}\sum_{<ij>}\boldsymbol{s}_{i}\boldsymbol{s}_{j}+\frac{\gamma}{kT}\boldsymbol{H}\cdot\sum_{i}\boldsymbol{s}_{i}}
\end{align}$$

The thermal expectation value of an observable \(O\) is then
$$\begin{align}
<O> &= \frac 1Z \int[\prod_{i}ds_{i}] \, O(s) \, e^{-\frac{1}{kT}E(s)}
\end{align}$$
 
At high temperatures the spins become essentially random and the
magnetisation dissappears.

## The Ising Model

The Ising model is a further simplification of the above. All the spins
are either $+1$ or $-1$. The paritition function then is
$$\begin{align}
Z & =\sum_{s_{i}=\pm1}e^{\beta\sum_{<ij>}s_{i}s_{j}+h\cdot\sum_{i}s_{i}}.
\end{align}$$
Here we use dimensionless couplings, $\beta=\frac{1}{kT}$ and
$h=\frac{\gamma H}{kT}$.

The Ising model has been solved in 1 and 2 dimensions **(citation
onsager).**


::::: {.card .bg-light}
**Example**

Let's implement the Ising model.
:::::

## Observables

We can measure the amount of magnetisation through the sum of the spins.
For an individual configuration

$$M=\frac{1}{V}\sum_{i}s_{i},$$

where V is the number of points on the lattice, the volume. We get the thermal average by integrating over the Boltzmann distribution:

$$\begin{align}
<M> &=\frac{1}{V} \frac 1Z \int[\prod_{i}ds_{i}]  e^{-\frac{1}{kT}E(s)} ( \sum_{i}s_{i}) 
\end{align}$$

This can also be expressed as a derivative of the partition function with respect to the external field $h$ 
$$\begin{align}
<M> &= \frac{\partial}{\partial h} \log(Z).
\end{align}$$
So the field \(h\) functions as a source for the magnetisation.

Similarly the energy is 
$$\begin{align}
<E> & = \frac 1Z \int[\prod_{i}ds_{i}] \, E(s) \, e^{-\beta E(s)} \\
&= -\frac{\partial}{\partial \beta} \log(Z)
\end{align}$$

Other intersting observables include
- the specific heat (heat capacity)
$$\begin{align}
\chi & = -\frac 1V \frac{\partial}{\partial \beta} <E> \\
& = \frac 1V \frac{\partial^2}{\partial^2 \beta} \log(Z) \\
& = -\frac 1V \frac{\partial}{\partial \beta} \frac 1Z \int[\prod_{i}ds_{i}] E(s) e^{-\beta E(s)} \\
& = \frac 1V \frac 1Z \int[\prod_{i}ds_{i}] E^2(s) e^{-\beta E(s)} - \frac 1V \frac 1{Z^2} \left(\int[\prod_{i}ds_{i}] E(s) e^{-\beta E(s)}\right)^2\\
&=\frac 1V \left( <E^2> - <E>^2 \right)
\end{align}$$

- The magnetic susceptibility
$$\begin{align}
\chi_M & = \frac 1V \frac{\partial}{\partial h} <M> = \frac 1V \frac{\partial^2}{\partial^2 h} \log(Z) \\
&= \frac 1V\left( <M^2> - <M>^2 \right)
\end{align}$$

- Correlation functions
$$\begin{align}
&C(\boldsymbol{x}-\boldsymbol{y}) = <s_{\boldsymbol{x}} s_{\boldsymbol{y}}> - <s_{\boldsymbol{x}}><s_{\boldsymbol{y}}>, \\
&\lim_{|\boldsymbol{x} - \boldsymbol{y}|\to\infty} C(\boldsymbol{x}-\boldsymbol{y}) = e^{-|\boldsymbol{x} - \boldsymbol{y}|/\xi},
\end{align}$$
where $\xi$ is the correlation length.

Deriving this from the partition function requires introducing an \(\boldsymbol{x}\)-dependent source \(h_\boldsymbol{x}\)
$$\begin{align}
Z & =\sum_{s_{i}=\pm1}e^{\beta\sum_{<ij>}s_{i}s_{j}+\sum_{i} h_i s_{i}}.
\end{align}$$
$$\begin{align}
&C(\boldsymbol{x}-\boldsymbol{y}) = \partial_\boldsymbol{x} \partial_\boldsymbol{y}
 \left . \log(Z) \right |_{h=0}
\end{align}$$



## Transfer matrices (Example of an exact solution)

Consider the 1D Ising model:
[image]
and assume periodic boundary conditions
$$\begin{align}
s_{x+L}=s_x
\end{align}$$

First we'll write the energy in a symmetric way between the neighbouring sites
$$\begin{align}
E &= \beta\sum_{x=1}^L s_x s_{x+1} +h\sum_{x=1}^L s_x\\
 &=\sum_{x=1}^L \left( \beta s_x s_x+1 + \frac 12 h s_x + s_{x+1} \right)
\end{align}$$

We'll define the \(2\times2\) transfer matrix
$$\begin{align}
T_{s,s'} = e^{\beta s s' + \frac 12 h(s+s')}.
\end{align}$$
Now the partition function can be written as
$$\begin{align}
z &= \sum_{\{s_x\}} T_{s_1,s_2} T_{s_2,s_3} \dots T_{s_{L-1},s_L} T_{s_L,s_1}\\
  &= \sum_{s_1} \left( T^L \right)_{s_1,s_1}\\
  &= Tr (T^L)
\end{align}$$

Writing the transfer matrix explicitly,
$$\begin{align}
T_{s,s'} = 
\left (\begin{matrix}
e^{\beta+h} & e^{-\beta} \\
e^{-\beta} & e^{\beta-h} 
\end{matrix} \right )
\end{align}$$
We can evaluate the trace by diagonalizing \(T\):
$$\begin{align}
& \det \left (\begin{matrix}
e^{\beta+h} -\lambda & e^{-\beta} \\
e^{-\beta} & e^{\beta-h} -\lambda
\end{matrix} \right )
= \left(e^{\beta+h}-\lambda\right) \left(e^{\beta-h}-\lambda\right) - e^{-2\beta}\\
& \lambda_{\pm} = e^\beta \left ( \cosh(h) \pm \sqrt{ \sinh^2(h)+e^{-4\beta} }\right )
\end{align}$$

$$\begin{align}
\log(Z) &= \log \left( \left (\begin{matrix}
\lambda_+ & 0 \\
0& \lambda_-
\end{matrix} \right )^L\right) 
= \log\left( \lambda_+^L + \lambda_-^L \right)\\
&= \log\left( \lambda_+^L \left(1+\left(\frac{\lambda_-}{\lambda_+}\right)^L \right ) \right)\\
&= \log\left( \lambda_+^L \right ) + \log\left ( 1+\left(\frac{\lambda_-}{\lambda_+}\right)^L \right)\\
\end{align}$$

In the thermodynamic limit, \(L\to\infty\),
$$\begin{equation}
\left(\frac{\lambda_-}{\lambda_+}\right)^L \to 0
\end{equation}$$
and 
$$\begin{align}
\log(Z) &= L\log \left( \cosh(h) + \sqrt{\sinh^2(h)+e^{-4\beta}} \right )
\end{align}$$

From here we can calculate the magnetisation as a function of \(h\)
$$\begin{align}
<M> &= \frac 1L \frac \partial {\partial h} \log(Z)
= \frac{\sinh(h) + \frac{\cosh(h) \sinh(h)}{\sqrt{\sinh^2(h)+e^{-4\beta}} }}{\cosh(h)+\sqrt{\sinh^2(h)+e^{-4\beta}} }\\
&= \frac{\sinh(h)}{\sqrt{\sinh(h)+e^{-4\beta}} }
\end{align}$$

So at \(h=0\) the magnetisation is zero, which is what we expect. At large \(\beta\), small temperature, it approaches one, which is also expected. Here is a schetch of its behaviour in general:

[Image]


## Phase transitions

### Symmetries

 - First order

 - Second order



-   Ising model

    -   Heath bath, Boltzmann distribution

    -   Magnetisation

-   Other spin models: Potts, U(1)

    -   Locally symmetric U(1) gauge model

-   Thermodynamics:

    -   Phase transitions, 1st and 2nd order, cross-over

Universality, Field theories
============================

-   Field theories, scalar field theory as an example

    -   Quantum field theory with a scalar field - derivation of
        discretized system

    -   Universality, critical points

    -   Thermal field theory

-   Quantum Monte Carlo,

Gauge fields
============

-   Representing gauge symmetry, back to U(1)

-   Updating a gauge field

-   Wilson loops

-   Perturbation theory, large and small coupling

Fermions
========

-   Fermion doubling, naive fermions

-   Wilson

    -   Symmetries

    -   O(a) improvement?

-   Staggered

    -   Symmetries, benefits and limitations

-   Measuring propagators

Hybrid Monte-Carlo
==================

-   The other problem with fermions, updating the gauge field

-   Molecular dynamics

-   Conjugate gradient

Project suggestions:
====================

-   Transition type and temperature in the U(N) model

-   Higgs model, transition

-   String tension in SU(3)

-   Counting naive fermions?

-   3D Thirring model (2D?)

$$<O>=\frac{\int_{U}Z(U)O}{\int_{U}Z(U)}$$
