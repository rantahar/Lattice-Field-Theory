.. Lattice Field Theories master file, created by
   sphinx-quickstart on Tue Jul 14 08:25:05 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**********************
Lattice Field Theories
**********************

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Motivation
==================

These lecture notes give an introduction to lattice field theory, a
powerful framework for solving quantum field theries from first
principles.
We approach the topic mainly from the point of view of QCD, going
through the building blocks necessary to simulate a model with fermions
with a gauge interaction.

Learning Objectives
==================

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
==================

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

.. math::
   \begin{align}
   E=-j\boldsymbol{s}_{1}\cdot\boldsymbol{s}_{2}
   \end{align}

and the energy of the whole system is

.. math::
   \begin{align}
   E=-J\sum_{<ij>}\boldsymbol{s}_{i}\boldsymbol{s}_{j}.
   \end{align}
The sum over
:math:`<ij>`
here counts pairs of neighbours. If the spins are coupled to a
magnetic field :math:`\boldsymbol{H}`, with the magnetic momentum :math:`\gamma`,
the energy is

.. math::
   \begin{align}
   E=-J\sum_{<ij>}\boldsymbol{s}_{i}\boldsymbol{s}_{j}-\gamma\boldsymbol{H}\cdot\sum_{i}\boldsymbol{s}_{i}.
   \end{align}

At small temperatures the energy will tend to a minimum and the spins
will align. If there is a magnetic field, they will align to the
direction of the field. Otherwise they will pick a random direction.

At a non-zero temperature :math:`T` the *configurations* of spins will follow
the Boltzmann distribution,

.. math::
   \begin{align}
   Z & =\int[\prod_{i}ds_{i}]e^{-\frac{1}{kT}E(s)}\\
     & =\int[\prod_{i}ds_{i}]e^{\frac{J}{kT}\sum_{<ij>}\boldsymbol{s}_{i}\boldsymbol{s}_{j}+\frac{\gamma}{kT}\boldsymbol{H}\cdot\sum_{i}\boldsymbol{s}_{i}}
   \end{align}

The thermal expectation value of an observable :math:`O` is then

.. math::
   \begin{align}
   <O> &= \frac 1Z \int[\prod_{i}ds_{i}] \, O(s) \, e^{-\frac{1}{kT}E(s)}
   \end{align}
 
At high temperatures the spins become essentially random and the
magnetisation dissappears.

The Ising Model
-----------

The Ising Model
==================

The Ising model is a further simplification of the above. All the spins
are either :math:`+1` or :math:`-1`. The paritition function then is

.. math::
   \begin{align}
   Z & =\sum_{s_{i}=\pm1}e^{\beta\sum_{<ij>}s_{i}s_{j}+h\cdot\sum_{i}s_{i}}.
   \end{align}
Here we use dimensionless couplings, :math:`\beta=\frac{1}{kT}` and
:math:`h=\frac{\gamma H}{kT}`.

The Ising model has been solved in 1 and 2 dimensions **(citation
onsager).**


::::: {.card .bg-light}
**Example**

Let's implement the Ising model.
:::::

Observables
-----------

We can measure the amount of magnetisation through the sum of the spins.
For an individual configuration


.. math::
   \begin{align}
   M=\frac{1}{V}\sum_{i}s_{i},
   \end{align}

where V is the number of points on the lattice, the volume. We get the thermal average by integrating over the Boltzmann distribution:


.. math::
   \begin{align}
   <M> &=\frac{1}{V} \frac 1Z \int[\prod_{i}ds_{i}]  e^{-\frac{1}{kT}E(s)} ( \sum_{i}s_{i}) 
   \end{align}

This can also be expressed as a derivative of the partition function with respect to the external field :math:`h` 

.. math::
   \begin{align}
   <M> &= \frac{1}{V} \frac{\partial}{\partial h} \log(Z).
   \end{align}
So the field :math:`h` functions as a source for the magnetisation.

Similarly the energy is 

.. math::
   \begin{align}
   <E> & = \frac 1Z \int[\prod_{i}ds_{i}] \, E(s) \, e^{-\beta E(s)} \\
   &= -\frac{\partial}{\partial \beta} \log(Z)
   \end{align}

Other intersting observables include
- the specific heat (heat capacity)

.. math::
   \begin{align}
   \chi & = -\frac 1V \frac{\partial}{\partial \beta} <E> \\
   & = \frac 1V \frac{\partial^2}{\partial^2 \beta} \log(Z) \\
   & = -\frac 1V \frac{\partial}{\partial \beta} \frac 1Z \int[\prod_{i}ds_{i}] E(s) e^{-\beta E(s)} \\
   & = \frac 1V \frac 1Z \int[\prod_{i}ds_{i}] E^2(s) e^{-\beta E(s)} - \frac 1V \frac 1{Z^2} \left(\int[\prod_{i}ds_{i}] E(s) e^{-\beta E(s)}\right)^2\\
   &=\frac 1V \left( <E^2> - <E>^2 \right)
   \end{align}

- The magnetic susceptibility

.. math::
   \begin{align}
   \chi_M & = \frac 1V \frac{\partial}{\partial h} <M> = \frac 1V \frac{\partial^2}{\partial^2 h} \log(Z) \\
   &= \frac 1V\left( <M^2> - <M>^2 \right)
   \end{align}

- Correlation functions

.. math::
   \begin{align}
   &C(\boldsymbol{x}-\boldsymbol{y}) = <s_{\boldsymbol{x}} s_{\boldsymbol{y}}> - <s_{\boldsymbol{x}}><s_{\boldsymbol{y}}>, \\
   &\lim_{|\boldsymbol{x} - \boldsymbol{y}|\to\infty} C(\boldsymbol{x}-\boldsymbol{y}) = e^{-|\boldsymbol{x} - \boldsymbol{y}|/\xi},
   \end{align}
where :math:`\xi` is the correlation length.

Deriving this from the partition function requires introducing an :math:`\boldsymbol{x}`-dependent source :math:`h_\boldsymbol{x}`

.. math::
   \begin{align}
   Z & =\sum_{s_{i}=\pm1}e^{\beta\sum_{<ij>}s_{i}s_{j}+\sum_{i} h_i s_{i}}.
   \end{align}

.. math::
   \begin{align}
   &C(\boldsymbol{x}-\boldsymbol{y}) = \partial_\boldsymbol{x} \partial_\boldsymbol{y}
     \left . \log(Z) \right |_{h=0}
   \end{align}



Transfer matrices (Example of an exact solution)
-----------

Consider the 1D Ising model:
[image]
and assume periodic boundary conditions

.. math::
   \begin{align}
   s_{x+L}=s_x
   \end{align}

First we'll write the energy in a symmetric way between the neighbouring sites

.. math::
   \begin{align}
   E &= \beta\sum_{x=1}^L s_x s_{x+1} +h\sum_{x=1}^L s_x\\
     &=\sum_{x=1}^L \left( \beta s_x s_x+1 + \frac 12 h s_x + s_{x+1} \right)
   \end{align}

We'll define the :math:`2\times2` transfer matrix

.. math::
   \begin{align}
   T_{s,s'} = e^{\beta s s' + \frac 12 h(s+s')}.
   \end{align}
Now the partition function can be written as

.. math::
   \begin{align}
   z &= \sum_{\{s_x\}} T_{s_1,s_2} T_{s_2,s_3} \dots T_{s_{L-1},s_L} T_{s_L,s_1}\\
     &= \sum_{s_1} \left( T^L \right)_{s_1,s_1}\\
     &= Tr (T^L)
   \end{align}

Writing the transfer matrix explicitly,

.. math::
   \begin{align}
   T_{s,s'} = 
   \left (\begin{matrix}
   e^{\beta+h} & e^{-\beta} \\
   e^{-\beta} & e^{\beta-h} 
   \end{matrix} \right )
   \end{align}
We can evaluate the trace by diagonalizing :math:`T`:

.. math::
   \begin{align}
   & \det \left (\begin{matrix}
   e^{\beta+h} -\lambda & e^{-\beta} \\
   e^{-\beta} & e^{\beta-h} -\lambda
   \end{matrix} \right )
   = \left(e^{\beta+h}-\lambda\right) \left(e^{\beta-h}-\lambda\right) - e^{-2\beta}\\
   & \lambda_{\pm} = e^\beta \left ( \cosh(h) \pm \sqrt{ \sinh^2(h)+e^{-4\beta} }\right )
   \end{align}


.. math::
   \begin{align}
   \log(Z) &= \log \left( \left (\begin{matrix}
   \lambda_+ & 0 \\
   0& \lambda_-
   \end{matrix} \right )^L\right) 
   = \log\left( \lambda_+^L + \lambda_-^L \right)\\
   &= \log\left( \lambda_+^L \left(1+\left(\frac{\lambda_-}{\lambda_+}\right)^L \right ) \right)\\
   &= \log\left( \lambda_+^L \right ) + \log\left ( 1+\left(\frac{\lambda_-}{\lambda_+}\right)^L \right)\\
   \end{align}

In the thermodynamic limit, :math:`L\to\infty`,
.. math::
   \begin{align}
   \left(\frac{\lambda_-}{\lambda_+}\right)^L \to 0
   \end{align}
and 

.. math::
   \begin{align}
   \log(Z) &= L\log \left( \cosh(h) + \sqrt{\sinh^2(h)+e^{-4\beta}} \right )
   \end{align}

From here we can calculate the magnetisation as a function of :math:`h`

.. math::
   \begin{align}
   <M> &= \frac 1L \frac \partial {\partial h} \log(Z)
   = \frac{\sinh(h) + \frac{\cosh(h) \sinh(h)}{\sqrt{\sinh^2(h)+e^{-4\beta}} }}{\cosh(h)+\sqrt{\sinh^2(h)+e^{-4\beta}} }\\
   &= \frac{\sinh(h)}{\sqrt{\sinh(h)+e^{-4\beta}} }
   \end{align}

So at :math:`h=0` the magnetisation is zero, which is what we expect. At large :math:`\beta`, small temperature, it approaches one, which is also expected. Here is a schetch of its behaviour in general:

[Image]




Phase transitions
-----------

### Global symmetries

The action

.. math::
   \begin{align}
   S = -\beta \sum_{<ij>} s_i \cdot s_j - h\cdot \sum_i s_i
   \end{align}
has global symmetries when :math:`h=0`. There are transformations :math:`M` that can be applied to all spins without changing the action. More precisely, :math:`S(s)` and :math:`ds` remain constant when

.. math::
   \begin{align}
   s_i \to s_i' = Ms_i, \textrm{ for all } i.
   \end{align}

- **Ising:**


.. math::
   \begin{align}
   s_i\to -s_i
   \end{align}

- **Pots Model:**


.. math::
   \begin{align}
   s_i\to (s_i+1)\bmod N_s
   \end{align}
(and other permutations)


- **O(N) Model:**


.. math::
   \begin{align}
   s_i\to Ms_i,
   \end{align}
where :math:`M` is a :math:`N\times N` matrix with

.. math::
   \begin{align}
   M^T M = I.
   \end{align}
So M is an orthogonal matrix: :math:`M^{-1} = M^T`. It belongs to the group of :math:`N\times N` orthogonal matrices, :math:`M\in O(N)`

The interaction term is invariant since

.. math::
   \begin{align}
   s_i \cdot s_j = (s_i)_\alpha (s_j)\alpha = \to M_{\alpha,\beta} (s_i)_\beta  M_{\beta,\gamma} (s_j)\gamma \\
   = s_i M^T M s_j = s_i s_j
   \end{align}

If :math:`h\neq 0`, the symmetry is "broken":

.. math::
   \begin{align} 
   h \cdot s_i \to (h)_\alpha M_{\alpha,\beta} (s_i)_\beta \neq h \cdot s_i
   \end{align}

We also need to check the integration measure :math:`ds`:

.. math::
   \begin{align}
   &\int_{|s_i|=1} d^Vs_i = \int d^Vs_i \delta(|s_i|-1)\\
   &\to \int d^Vs'_i \delta(|s'_i|-1) =
   \int d^Vs_i \left \| \frac{ds'}{ds} \right \| \delta(|Ms'_i|-1)
   \\
   &= \int d^Vs_i | \det(M) | \delta(|Ms'_i|-1)\\
   &= \int d^Vs_i \delta(|Ms'_i|-1)
   \end{align}

So the measure is also invariant, and the model is invariant at :math:`h\neq 0`.


### Symmetry breaking

Consider the model at :math:`h=0` and finite :math:`V`. Since the model has a global :math:`O(N)` symmetry, it is symmetric under

.. math::
   \begin{align}
   s_i \to M s_i = -s_i (-I \in O(N)).
   \end{align}
However the magnetization

.. math::
   \begin{align}
   <M> = \frac{\int [ds] \sum_i s_i e^{-S}}{Z}
   \end{align}
is not symmetric,

.. math::
   \begin{align}
   <M> \to -<M>.
   \end{align}
Therefore we always have

.. math::
   \begin{align}
   <M> =0.
   \end{align}

- On a finite lattice the symmetry is *restored* and the model is in a *disordered phase*.

- If :math:`h\neq` the symmetry is explicitly broken and :math:`<M>\neq 0`

Non-trivial symmetry breaking happens in the thermodynamic limit, :math:`V\to 0`. 
The symmetry is spontaneously broken if

.. math::
   \begin{align}
   \lim_{h\to 0} \left[ \lim_{V\to\infty} <M> \right ]
   \end{align}
The order of the limits is important here. If the limit :math:`h\to0` is taken too quickly the magnetisation will approach :math:`0`.

- The 1D Ising model the symmetry is *not* spontaneously broken. 

.. math::
   \begin{align}
   \lim_{N\to\infty} <M> = \sinh(h) \times \frac{\sinh(h)}{\sqrt{\sinh^2 + e^{-4\beta}}}
   \end{align}

- At :math:`D>1` the symmetry is broken at :math:`T>T_c`, or :math:`\beta < \beta_c`.


### Phase transitions

In many models we have a broken symmetry at :math:`\beta > \beta_c` and a restored symmetry at :math:`\beta < \beta_c`. This means there is a phase transition at :math:`\beta=\beta_c`

 - **First order**
 One or more of the first derivatives of the gree energy :math:`F=-\log(Z)` os discontinuous:

.. math::
   \begin{align}
   <E> = \frac{\partial}{\partial\beta} F
   \end{align}

.. math::
   \begin{align}
   <M> = \frac{\partial}{\partial h} F
   \end{align}

[sketch]

The jump in energy is known as the latent heat

.. math::
   \begin{align} 
   \frac 1V \Delta E = \lim_{\beta\to_-\beta_c}<E> - \lim_{\beta\to_+\beta_c}<E> = \frac{E_- - E_+}{V}
   \end{align}

How does this work on the lattice? The transition is not instantaneous,
but get's smoothed over a small span of temperatures.
The derivative of the energy in fact grow with the volume.

.. math::
   \begin{align} 
   \chi = \frac 1V \left<(E-<E>)^2\right> 
   \end{align}
At :math:`\beta=\beta_c`, :math:`<E>\approx \frac 12 (E_+ + E_-)`, so

.. math::
   \begin{align} 
   \chi \approx \frac 1V \frac{\Delta E^2}{4}
   = V \frac 14 \left( \frac{\Delta E}{V} \right) \sim V
   \end{align}

In a first order transition the two phases can coexist, such as ice and water. The average energy density in this state is between the two phases.


 - **Second order**

No discontinuity in the first derivative of the free energy, 
but there is a discontinuity in the second derivative.
This is the case in spin models.

.. math::
   \begin{align} 
   \frac{\partial}{\partial_h}<M> \neq 0
   \end{align}

[scketch]

##### Critical Phenomena:

The correlation length :math:`\xi` diverges exponentially at :math:`\beta_c`.
Structures scale to all distances:

Writing :math:`\tau = \beta-\beta_c`:

.. math::
   \begin{align}
   \chi &\sim |\tau |^{-\alpha} \\
   \chi_M &\sim |\tau |^{-\gamma} \\
   \xi &\sim |\tau |^{-\nu} 
   \end{align}


.. math::
   \begin{align}
   \frac{<M>}{V} &= 0 \textrm{ at } \beta \leq \beta_c \textrm{ and }\\
   \frac{<M>}{V} &\sim |\tau|^\delta \textrm{ at } \beta > \beta_c
   \end{align}

The critical exponents are characteristic to the symmetries and dimensionality of the model.
This is an important property of higher order transitions known as universality.
It allows us to construct lattice models of continuous systems.
More on this later.

 - Potts model in 2D (Including Ising)
  Has a phase transition at :math:`\beta=\log(1+\sqrt(q))`. 
  It is second order when :math:`q\leq 4` and first order otherwise.
  This is a discrete symmetry, in 2D continuous symmetries do not break (Mermin-Wagner-Coleman theorem).

 - O(N) model in 2D
  No symmetry breaking transition due to Mermin-Wagner-Coleman. 

 - XY model in 2D
  Has a special Berezinskii–Kosterlitz–Thouless transition, :math:`\infty` order with symmetry restored on both sides.

 - O(N) in 3D
   Has a 2nd order transition. The critical exponents have been determined numerically.

 - Potts model in 3D
   First order when :math:`q\geq 3`, otherwise second.

 - O(N) in 4 or more dimensions
   Second order transition with *mean field* exponents. These can be calculated analytically.


In the O(N), Ising and Potts models, there is also a first order transition
when :math:`\beta > \beta_c` :math:`h\neq 0`, if we change :math:`h` continuously accross :math:`0`.

-  **Crossover**
  
Crossover is a term for a transition, where no symmetry is broken and / or 
there is no discontinuity in the derivative of the free energy.
The transition is continuous and there is no well defined critical
temperature, but the two phases are nevertheless distinct.



Field theories
============================

Now we will approach quantum field theories using *Feynman's path integral*. [Phys. Rev. Mod. Phys. 20, 1948].
In this representation, expecation values are calculated as


.. math::
   \begin{align}
   &<O> = \frac{1}{Z} \int \left [ \prod_x d\phi(x) \right ] O(\phi) e^{\frac{i}{\hbar} S(\phi)} \label{pathintegral}\\
   & \textrm{ where } Z= \int \left [ \prod_x d\phi(x) \right ]e^{\frac{i}{\hbar} S(\phi)}\\
   & \textrm{ and } S=\int d^4x \mathcal L (\phi, \partial_t \phi) 
   \end{align}

Or using natural units :math:`\hbar = 1`

.. math::
   \begin{align}
   Z = \left [ \prod_x d\phi(x) \right ] e^{i S(\phi)}
   \end{align}

This is similar to the representation of thermodynamics used above. 
We can write observables using source fields,

.. math::
   \begin{align}
   Z(J) &= \left [ \prod_x d\phi(x) \right ] e^{i \left (S(\phi) + J(x) \phi(x) \right ) }\\
   <\phi(x)> &= \left . \frac{\partial}{i\partial J(x)}\right |_{J=0} \log( Z(J) ) \\
   <\phi(x) \phi(y)> &= \left . \frac{\partial}{i\partial J(x)} \frac{\partial}{i\partial J(y)}\right |_{J=0} \log( Z(J) ) \\
   &= \frac 1Z \int \left [ d\phi \right ] \phi(x) \phi(y) e^{iS} - <\phi(x)>^2
   \end{align}


Now, since the fields are defined at all space-time locations, the integral measure

.. math::
   \begin{align}
   \prod_x d\phi(x) 
   \end{align}
is not well defined and needs to be regularized. This is a general problem with functional integrals.
Lattice field theory is a conceptually simple renormalization method:
we divide the volume into a lattice of discrete points :math:`x\in aZ^4` and study a system with
a finite volume :math:`V`. 
Since the intgeral is now finite, we can in principle calculate is directly (brute force, with supercomputers,)
and get fully non-perturbative results.

The full theory is recovered by taking the infinite volume and continuum limit :math:`v\to \infty, a\to0`.
The order of the limits is important here, just like for the spin models.

In practice the dimensionality of the integral grows quickly when increasing the volume and decreasing the lattice spacing.
In most cases the integral can be calculated directly only for lattice sizes that are practically useless.

Instead, we should use Montecarlo methods.
The problem here is the complex, unimodular weight, :math:`\exp(iS)`.
Every configuration :math:`\{\phi\}` contributes with the same magnitude and result depends on cancellations between configurations.
However, this is (mostly) solved in the imaginary time formalism of thermal field theory.


### The imaginary time path integral

Let's consider a scalar field theory is Minkowsky spacetime (the field :math:`\phi` could also represent a more complicated set of fields). Given the action

.. math::
   \begin{align}
   S = \int d^3 dt \mathcal L(\phi,\partial_t \phi) = \int d^3 dt \left [ \frac 12 \partial_\mu \phi \partial^\mu\phi - V(\phi) \right ]
   \end{align}
The classical Hamiltonian is obtained by a Legendre transformation

.. math::
   \begin{align}
   H &= \int d^3xdt\left [ \pi\dot\phi-\mathcal L \right ],
   \end{align}
where :math:`\pi = \delta \mathcal L/\delta\dot\phi` is the canonical momentum, and

.. math::
   \begin{align}
   H &= \int d^3xdt\left [ \pi^2 +\frac 12 (\partial_i\phi)^2 + V(\phi) \right ].
   \end{align}

In quantum field theory, we consider the Hilbert space of states :math:`|\phi>`, :math:`|\pi>`
and the Hamiltonian operator

.. math::
   \begin{align}
   \hat H |\phi,\pi> = H |\phi,\pi>.
   \end{align}
Let's also define the field operators

.. math::
   \begin{align}
   &\hat \phi({\bf x}) |\phi> = \phi({\bf x}) |\phi>\\
   &\hat \pi({\bf x}) |\pi> = \pi({\bf x}) |\pi>\\
   &\int \left[\prod_x d\phi({\bf x})\right] |\phi><\phi| = 1\\
   &\int \left[\prod_x \frac {d\pi({\bf x})}{2\pi} \right] |\pi><\pi| = 1\\
   &[\hat\phi({\bf x}),\hat\phi({\bf x}')] = -i\delta^3({\bf x} - {\bf x}') \\
   &<\phi|\pi> = e^{i\int d^3xdt \pi({\bf x})\phi({\bf x})}
   \end{align}
In this representation the time evolution operator is 

.. math::
   \begin{align}
   U(t)=e^{i\hat H t}
   \end{align}

In this representation, we can write expectation values in using the
partition function,

.. math::
   \begin{align}
   Z = \textrm{Tr} e^{i\hat H t} = \int \left[d\phi \right] \left<\phi\left| e^{i \hat H t} \right|\phi\right>.
   \end{align}
From here we could derive the Feynman path integral representation :math:`\ref{pathintegral}`.
Here we will follow the derivation,
evolving the a field configuration :math:`<\phi|` by small time steps :math:`\delta t` and 
taking the limit :math:`\delta t\to 0`,
but in the imaginary time formalism.

In equilibrium at a finite temperature :math:`T`, a quantum field theory follows the Gibbs ensemble:

.. math::
   \begin{align}
   Z = \textrm{Tr} e^{-\hat H/T} = \int \left[d\phi \right] \left<\phi\left| e^{-\hat H/T} \right|\phi\right>.
   \end{align}
This is formally similar to the real time partition function above, with the replacement

.. math::
   \begin{align}
   \tau = it = 1/T
   \end{align}
and is equaivalent to quantum field theory in *Euclidean* spacetime.
The Hamiltonian and the Hilbert space remain unchanged.

It is convenient to start by discretizing the space.
We will do this in any case, since we want to end up with discrete spacetime, but could be done later.

.. math::
   \begin{align}
   &{\bf x} = a {\bf n}, n_i = 0...N_s,\\
   &\int d^3x \to a^3 \sum_x\\
   &\partial_i\phi \to \frac 1a \left[ \phi({\bf x}+{\bf e}_ia) -\phi(x) \right] = \Delta_i\phi({\bf x})
   \end{align}





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