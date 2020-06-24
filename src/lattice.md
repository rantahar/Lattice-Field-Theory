---
title: Lattice Field Theory
papersize: A4
---

Motivation
==========

These lecture notes give an introduction to lattice field theory, a
powerful framework for solving quantum field theries from first
principles.
We approach the topic mainly from the point of view of QCD, going
through the building blocks necessary to simulate a model with fermions
with a gauge interaction.

Learning Objectives
===================

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


Spin models
===========

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

$$\begin{aligned}
Z & =\int[\prod_{i}ds_{i}]e^{-\frac{1}{kT}E(s)}\\
 & =\int[\prod_{i}ds_{i}]e^{\frac{J}{kT}\sum_{<ij>}\boldsymbol{s}_{i}\boldsymbol{s}_{j}+\frac{\gamma}{kT}\boldsymbol{H}\cdot\sum_{i}\boldsymbol{s}_{i}}
\end{aligned}$$

The thermal expectation value of an observable \(O\) is then
$$\begin{aligned}
<O> &= \frac 1Z \int[\prod_{i}ds_{i}] \, O(s) \, e^{-\frac{1}{kT}E(s)}
\end{aligned}$$
 
At high temperatures the spins become essentially random and the
magnetisation dissappears.

### The Ising Model

The Ising model is a further simplification of the above. All the spins
are either $+1$ or $-1$. The paritition function then is
$$\begin{aligned}
Z & =\sum_{s_{i}=\pm1}e^{\beta\sum_{<ij>}s_{i}s_{j}+h\cdot\sum_{i}s_{i}}.\end{aligned}$$
Here we use dimensionless couplings, $\beta=\frac{1}{kT}$ and
$h=\frac{\gamma H}{kT}$.

The Ising model has been solved in 1 and 2 dimensions **(citation
onsager).**


::::: {.card .bg-light}
**Example**

Let's implement the Ising model.
:::::

### Observables

We can measure the amount of magnetisation through the sum of the spins.
For an individual configuration

$$M=\frac{1}{V}\sum_{i}s_{i},$$

where V is the number of points on the lattice, the volume. We get the thermal average by integrating over the Boltzmann distribution:

$$\begin{aligned}
<M> &=\frac{1}{V} \frac 1Z \int[\prod_{i}ds_{i}]  e^{-\frac{1}{kT}E(s)} ( \sum_{i}s_{i}) 
\end{aligned}$$

This can also be expressed as a derivative of the partition function with respect to the external field $h$ 
$$\begin{aligned}
<M> &= \frac{\partial}{\partial h} \log(Z).
\end{aligned}$$
So the field \(h\) functions as a source for the magnetisation.

Similarly the energy is 
$$\begin{aligned}
<E> & = \frac 1Z \int[\prod_{i}ds_{i}] \, E(s) \, e^{-\beta E(s)} \\
&= -\frac{\partial}{\partial \beta} \log(Z)
\end{aligned}$$

Other intersting observables include
- the specific heat (heat capacity)
$$\begin{aligned}
\chi & = -\frac 1V \frac{\partial}{\partial \beta} <E> \\
& = \frac 1V \frac{\partial^2}{\partial^2 \beta} \log(Z) \\
& = -\frac 1V \frac{\partial}{\partial \beta} \frac 1Z \int[\prod_{i}ds_{i}] E(s) e^{-\beta E(s)} \\
& = \frac 1V \frac 1Z \int[\prod_{i}ds_{i}] E^2(s) e^{-\beta E(s)} - \frac 1V \frac 1{Z^2} \left(\int[\prod_{i}ds_{i}] E(s) e^{-\beta E(s)}\right)^2\\
&=\frac 1V \left( <E^2> - <E>^2 \right)
\end{aligned}$$

- The magnetic susceptibility
$$\begin{aligned}
\chi & = \frac 1V \frac{\partial}{\partial h} <M> = \frac 1V \frac{\partial^2}{\partial^2 h} \log(Z) \\
&= \frac 1V\left( <M^2> - <M>^2 \right)
\end{aligned}$$

- Correlation functions
$$\begin{aligned}
&C(\bold x-\bold y) = <s_{\bold x} s_{\bold y}> - <s_{\bold x}><s_{\bold y}>, \\
&\lim_{\to\infty} C(\bold x-\bold y) = e^{-|\bold x - \bold y|/\xi},
\end{aligned}$$
where $\xi$ is the correlation length.

###Phase transitions


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
