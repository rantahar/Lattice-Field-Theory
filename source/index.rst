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
   E=-j\mathbf{s}_{1}\cdot\mathbf{s}_{2}
   :label: 

and the energy of the whole system is

.. math::
   E=-J\sum_{<ij>}\mathbf{s}_{i}\mathbf{s}_{j}.
   :label:
The sum over
:math:`<ij>`
here counts pairs of neighbours. If the spins are coupled to a
magnetic field :math:`\mathbf{H}`, with the magnetic momentum :math:`\gamma`,
the energy is

.. math::
   E=-J\sum_{<ij>}\mathbf{s}_{i}\mathbf{s}_{j}-\gamma\mathbf{H}\cdot\sum_{i}\mathbf{s}_{i}.
   :label:

At small temperatures the energy will tend to a minimum and the spins
will align. If there is a magnetic field, they will align to the
direction of the field. Otherwise they will pick a random direction.

At a non-zero temperature :math:`T` the *configurations* of spins will follow
the Boltzmann distribution,

.. math::
   Z & =\int[\prod_{i}ds_{i}]e^{-\frac{1}{kT}E(s)}\\
     & =\int[\prod_{i}ds_{i}]e^{\frac{J}{kT}\sum_{<ij>}\mathbf{s}_{i}\mathbf{s}_{j}+\frac{\gamma}{kT}\mathbf{H}\cdot\sum_{i}\mathbf{s}_{i}}
   :label:

The thermal expectation value of an observable :math:`O` is then

.. math::
   <O> &= \frac 1Z \int[\prod_{i}ds_{i}] \, O(s) \, e^{-\frac{1}{kT}E(s)}
   :label:
 
At high temperatures the spins become essentially random and the
magnetisation dissappears.

The Ising Model
-----------


The Ising model is a further simplification of the above. All the spins
are either :math:`+1` or :math:`-1`. The paritition function then is

.. math::
   Z & =\sum_{s_{i}=\pm1}e^{\beta\sum_{<ij>}s_{i}s_{j}+h\cdot\sum_{i}s_{i}}.
   :label:
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
   M=\frac{1}{V}\sum_{i}s_{i},
   :label:

where V is the number of points on the lattice, the volume. We get the thermal average by integrating over the Boltzmann distribution:


.. math::
   <M> &=\frac{1}{V} \frac 1Z \int[\prod_{i}ds_{i}]  e^{-\frac{1}{kT}E(s)} ( \sum_{i}s_{i}) 
   :label:

This can also be expressed as a derivative of the partition function with respect to the external field :math:`h` 

.. math::
   <M> &= \frac{1}{V} \frac{\partial}{\partial h} \log(Z).
   :label:
So the field :math:`h` functions as a source for the magnetisation.

Similarly the energy is 

.. math::
   <E> & = \frac 1Z \int[\prod_{i}ds_{i}] \, E(s) \, e^{-\beta E(s)} \\
   &= -\frac{\partial}{\partial \beta} \log(Z)
   :label:

Other intersting observables include
- the specific heat (heat capacity)

.. math::
   \chi & = -\frac 1V \frac{\partial}{\partial \beta} <E> \\
   & = \frac 1V \frac{\partial^2}{\partial^2 \beta} \log(Z) \\
   & = -\frac 1V \frac{\partial}{\partial \beta} \frac 1Z \int[\prod_{i}ds_{i}] E(s) e^{-\beta E(s)} \\
   & = \frac 1V \frac 1Z \int[\prod_{i}ds_{i}] E^2(s) e^{-\beta E(s)} - \frac 1V \frac 1{Z^2} \left(\int[\prod_{i}ds_{i}] E(s) e^{-\beta E(s)}\right)^2\\
   &=\frac 1V \left( <E^2> - <E>^2 \right)
   :label:

- The magnetic susceptibility

.. math::
   \chi_M & = \frac 1V \frac{\partial}{\partial h} <M> = \frac 1V \frac{\partial^2}{\partial^2 h} \log(Z) \\
   &= \frac 1V\left( <M^2> - <M>^2 \right)
   :label:

- Correlation functions

.. math::
   &C(\mathbf{x}-\mathbf{y}) = <s_{\mathbf{x}} s_{\mathbf{y}}> - <s_{\mathbf{x}}><s_{\mathbf{y}}>, \\
   &\lim_{|\mathbf{x} - \mathbf{y}|\to\infty} C(\mathbf{x}-\mathbf{y}) = e^{-|\mathbf{x} - \mathbf{y}|/\xi},
   :label:
where :math:`\xi` is the correlation length.

Deriving this from the partition function requires introducing an :math:`\mathbf{x}`-dependent source :math:`h_\mathbf{x}`

.. math::
   Z & =\sum_{s_{i}=\pm1}e^{\beta\sum_{<ij>}s_{i}s_{j}+\sum_{i} h_i s_{i}}.
   :label:

.. math::
   &C(\mathbf{x}-\mathbf{y}) = \partial_\mathbf{x} \partial_\mathbf{y}
     \left . \log(Z) \right |_{h=0}
   :label:



Transfer matrices (Example of an exact solution)
-----------

Consider the 1D Ising model:
[image]
and assume periodic boundary conditions

.. math::
   s_{x+L}=s_x
   :label:

First we'll write the energy in a symmetric way between the neighbouring sites

.. math::
   E &= \beta\sum_{x=1}^L s_x s_{x+1} +h\sum_{x=1}^L s_x\\
     &=\sum_{x=1}^L \left( \beta s_x s_x+1 + \frac 12 h s_x + s_{x+1} \right)
   :label:

We'll define the :math:`2\times2` transfer matrix

.. math::
   T_{s,s'} = e^{\beta s s' + \frac 12 h(s+s')}.
   :label:
Now the partition function can be written as

.. math::
   z &= \sum_{\{s_x\}} T_{s_1,s_2} T_{s_2,s_3} \dots T_{s_{L-1},s_L} T_{s_L,s_1}\\
     &= \sum_{s_1} \left( T^L \right)_{s_1,s_1}\\
     &= Tr (T^L)
   :label:

Writing the transfer matrix explicitly,

.. math::
   T_{s,s'} = 
   \left (\begin{matrix}
   e^{\beta+h} & e^{-\beta} \\
   e^{-\beta} & e^{\beta-h} 
   \end{matrix} \right )
   :label:
We can evaluate the trace by diagonalizing :math:`T`:

.. math::
   & \det \left (\begin{matrix}
   e^{\beta+h} -\lambda & e^{-\beta} \\
   e^{-\beta} & e^{\beta-h} -\lambda
   \end{matrix} \right )
   = \left(e^{\beta+h}-\lambda\right) \left(e^{\beta-h}-\lambda\right) - e^{-2\beta}\\
   & \lambda_{\pm} = e^\beta \left ( \cosh(h) \pm \sqrt{ \sinh^2(h)+e^{-4\beta} }\right )
   :label:


.. math::
   \log(Z) &= \log \left( \left (\begin{matrix}
   \lambda_+ & 0 \\
   0& \lambda_-
   \end{matrix} \right )^L\right) 
   = \log\left( \lambda_+^L + \lambda_-^L \right)\\
   &= \log\left( \lambda_+^L \left(1+\left(\frac{\lambda_-}{\lambda_+}\right)^L \right ) \right)\\
   &= \log\left( \lambda_+^L \right ) + \log\left ( 1+\left(\frac{\lambda_-}{\lambda_+}\right)^L \right)\\
   :label:

In the thermodynamic limit, :math:`L\to\infty`,

.. math::
   \left(\frac{\lambda_-}{\lambda_+}\right)^L \to 0
   :label:
and 

.. math::
   \log(Z) &= L\log \left( \cosh(h) + \sqrt{\sinh^2(h)+e^{-4\beta}} \right )
   :label:

From here we can calculate the magnetisation as a function of :math:`h`

.. math::
   <M> &= \frac 1L \frac \partial {\partial h} \log(Z)
   = \frac{\sinh(h) + \frac{\cosh(h) \sinh(h)}{\sqrt{\sinh^2(h)+e^{-4\beta}} }}{\cosh(h)+\sqrt{\sinh^2(h)+e^{-4\beta}} }\\
   &= \frac{\sinh(h)}{\sqrt{\sinh(h)+e^{-4\beta}} }
   :label:

So at :math:`h=0` the magnetisation is zero, which is what we expect. At large :math:`\beta`, small temperature, it approaches one, which is also expected. Here is a schetch of its behaviour in general:

[Image]




Phase transitions
-----------


**Global symmetries**


The action

.. math::
   S = -\beta \sum_{<ij>} s_i \cdot s_j - h\cdot \sum_i s_i
   :label:
has global symmetries when :math:`h=0`. There are transformations :math:`M` that can be applied to all spins without changing the action. More precisely, :math:`S(s)` and :math:`ds` remain constant when

.. math::
   s_i \to s_i' = Ms_i, \textrm{ for all } i.
   :label:

- **Ising:**


.. math::
   s_i\to -s_i
   :label:

- **Pots Model:**


.. math::
   s_i\to (s_i+1)\bmod N_s
   :label:
(and other permutations)


- **O(N) Model:**


.. math::
   s_i\to Ms_i,
   :label:
where :math:`M` is a :math:`N\times N` matrix with

.. math::
   M^T M = I.
   :label:
So M is an orthogonal matrix: :math:`M^{-1} = M^T`. It belongs to the group of :math:`N\times N` orthogonal matrices, :math:`M\in O(N)`

The interaction term is invariant since

.. math::
   s_i \cdot s_j = (s_i)_\alpha (s_j)\alpha = \to M_{\alpha,\beta} (s_i)_\beta  M_{\beta,\gamma} (s_j)\gamma \\
   = s_i M^T M s_j = s_i s_j
   :label:

If :math:`h\neq 0`, the symmetry is "broken":

.. math::
   h \cdot s_i \to (h)_\alpha M_{\alpha,\beta} (s_i)_\beta \neq h \cdot s_i
   :label:

We also need to check the integration measure :math:`ds`:

.. math::
   &\int_{|s_i|=1} d^Vs_i = \int d^Vs_i \delta(|s_i|-1)\\
   &\to \int d^Vs'_i \delta(|s'_i|-1) =
   \int d^Vs_i \left \| \frac{ds'}{ds} \right \| \delta(|Ms'_i|-1)
   \\
   &= \int d^Vs_i | \det(M) | \delta(|Ms'_i|-1)\\
   &= \int d^Vs_i \delta(|Ms'_i|-1)
   :label:

So the measure is also invariant, and the model is invariant at :math:`h\neq 0`.


**Symmetry breaking**


Consider the model at :math:`h=0` and finite :math:`V`. Since the model has a global :math:`O(N)` symmetry, it is symmetric under

.. math::
   s_i \to M s_i = -s_i (-I \in O(N)).
   :label:
However the magnetization

.. math::
   <M> = \frac{\int [ds] \sum_i s_i e^{-S}}{Z}
   :label:
is not symmetric,

.. math::
   <M> \to -<M>.
   :label:
Therefore we always have

.. math::
   <M> =0.
   :label:

- On a finite lattice the symmetry is *restored* and the model is in a *disordered phase*.

- If :math:`h\neq` the symmetry is explicitly broken and :math:`<M>\neq 0`

Non-trivial symmetry breaking happens in the thermodynamic limit, :math:`V\to 0`. 
The symmetry is spontaneously broken if

.. math::
   \lim_{h\to 0} \left[ \lim_{V\to\infty} <M> \right ]
   :label:
The order of the limits is important here. If the limit :math:`h\to0` is taken too quickly the magnetisation will approach :math:`0`.

- The 1D Ising model the symmetry is *not* spontaneously broken. 

.. math::
   \lim_{N\to\infty} <M> = \sinh(h) \times \frac{\sinh(h)}{\sqrt{\sinh^2 + e^{-4\beta}}}
   :label:

- At :math:`D>1` the symmetry is broken at :math:`T>T_c`, or :math:`\beta < \beta_c`.


**Phase transitions**


In many models we have a broken symmetry at :math:`\beta > \beta_c` and a restored symmetry at :math:`\beta < \beta_c`. This means there is a phase transition at :math:`\beta=\beta_c`

 - **First order**
 One or more of the first derivatives of the gree energy :math:`F=-\log(Z)` os discontinuous:

.. math::
   <E> = \frac{\partial}{\partial\beta} F
   :label:

.. math::
   <M> = \frac{\partial}{\partial h} F
   :label:

[sketch]

The jump in energy is known as the latent heat

.. math::
   \frac 1V \Delta E = \lim_{\beta\to_-\beta_c}<E> - \lim_{\beta\to_+\beta_c}<E> = \frac{E_- - E_+}{V}
   :label:

How does this work on the lattice? The transition is not instantaneous,
but get's smoothed over a small span of temperatures.
The derivative of the energy in fact grow with the volume.

.. math::
   \chi = \frac 1V \left<(E-<E>)^2\right> 
   :label:
At :math:`\beta=\beta_c`, :math:`<E>\approx \frac 12 (E_+ + E_-)`, so

.. math::
   \chi \approx \frac 1V \frac{\Delta E^2}{4}
   = V \frac 14 \left( \frac{\Delta E}{V} \right) \sim V
   :label:

In a first order transition the two phases can coexist, such as ice and water. The average energy density in this state is between the two phases.


 - **Second order**

No discontinuity in the first derivative of the free energy, 
but there is a discontinuity in the second derivative.
This is the case in spin models.

.. math::
   \frac{\partial}{\partial_h}<M> \neq 0
   :label:

[scketch]

Critical Phenomena:
""""""""""""""

The correlation length :math:`\xi` diverges exponentially at :math:`\beta_c`.
Structures scale to all distances:

Writing :math:`\tau = \beta-\beta_c`:

.. math::
   \chi &\sim |\tau |^{-\alpha} \\
   \chi_M &\sim |\tau |^{-\gamma} \\
   \xi &\sim |\tau |^{-\nu} 
   :label:


.. math::
   \frac{<M>}{V} &= 0 \textrm{ at } \beta \leq \beta_c \textrm{ and }\\
   \frac{<M>}{V} &\sim |\tau|^\delta \textrm{ at } \beta > \beta_c
   :label:

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
   &<O> = \frac{1}{Z} \int \left [ \prod_x d\phi(x) \right ] O(\phi) e^{\frac{i}{\hbar} S(\phi)} \\
   & \textrm{ where } Z= \int \left [ \prod_x d\phi(x) \right ]e^{\frac{i}{\hbar} S(\phi)}\\
   & \textrm{ and } S=\int d^4x \mathcal L (\phi, \partial_t \phi) 
   :label: pathintegral

Or using natural units :math:`\hbar = 1`

.. math::
   Z = \left [ \prod_x d\phi(x) \right ] e^{i S(\phi)}
   :label:

This is similar to the representation of thermodynamics used above. 
We can write observables using source fields,

.. math::
   Z(J) &= \left [ \prod_x d\phi(x) \right ] e^{i \left (S(\phi) + J(x) \phi(x) \right ) }\\
   <\phi(x)> &= \left . \frac{\partial}{i\partial J(x)}\right |_{J=0} \log( Z(J) ) \\
   <\phi(x) \phi(y)> &= \left . \frac{\partial}{i\partial J(x)} \frac{\partial}{i\partial J(y)}\right |_{J=0} \log( Z(J) ) \\
   &= \frac 1Z \int \left [ d\phi \right ] \phi(x) \phi(y) e^{iS} - <\phi(x)>^2
   :label:


Now, since the fields are defined at all space-time locations, the integral measure

.. math::
   \prod_x d\phi(x) 
   :label:
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


**The imaginary time path integral**


Let's consider a scalar field theory is Minkowsky spacetime (the field :math:`\phi` could also represent a more complicated set of fields). Given the action

.. math::
   S = \int d^3 dt \mathcal L(\phi,\partial_t \phi) = \int d^3 dt \left [ \frac 12 \partial_\mu \phi \partial^\mu\phi - V(\phi) \right ]
   :label:
The classical Hamiltonian is obtained by a Legendre transformation

.. math::
   H &= \int d^3xdt\left [ \pi\dot\phi-\mathcal L \right ],
   :label:
where :math:`\pi = \delta \mathcal L/\delta\dot\phi` is the canonical momentum, and

.. math::
   H &= \int d^3xdt\left [ \pi^2 +\frac 12 (\partial_i\phi)^2 + V(\phi) \right ].
   :label:

In quantum field theory, we consider the Hilbert space of states :math:`|\phi>`, :math:`|\pi>`
and the Hamiltonian operator

.. math::
   \hat H |\phi,\pi> = H |\phi,\pi>.
   :label:
Let's also define the field operators

.. math::
   &\hat \phi({\bf x}) |\phi> = \phi({\bf x}) |\phi>\\
   &\hat \pi({\bf x}) |\pi> = \pi({\bf x}) |\pi>\\
   &\int \left[\prod_x d\phi({\bf x})\right] |\phi><\phi| = 1\\
   &\int \left[\prod_x \frac {d\pi({\bf x})}{2\pi} \right] |\pi><\pi| = 1\\
   &[\hat\phi({\bf x}),\hat\phi({\bf x}')] = -i\delta^3({\bf x} - {\bf x}') \\
   &<\phi|\pi> = e^{i\int d^3xdt \pi({\bf x})\phi({\bf x})}
   :label:
In this representation the time evolution operator is 

.. math::
   U(t)=e^{i\hat H t}
   :label:

In this representation, we can write expectation values in using the
partition function,

.. math::
   Z = \textrm{Tr} e^{i\hat H t} = \int d\phi \left<\phi\left| e^{i \hat H t} \right|\phi\right>.
   :label:
From here we could derive the Feynman path integral representation :eq:`pathintegral`.
Here we will follow the derivation,
evolving the a field configuration :math:`<\phi|` by small time steps :math:`\delta t` and 
taking the limit :math:`\delta t\to 0`,
but in the imaginary time formalism.

In equilibrium at a finite temperature :math:`T`, a quantum field theory follows the Gibbs ensemble:

.. math::
   Z = \textrm{Tr} e^{-\hat H/T} = \int d\phi \left<\phi\left| e^{-\hat H/T} \right|\phi\right>.
   :label:
This is formally similar to the real time partition function above, with the replacement

.. math::
   \tau = it = 1/T
   :label:
and is equivalent to quantum field theory in *Euclidean* spacetime.
The Hamiltonian and the Hilbert space remain unchanged.

It is convenient to start by discretizing the space.
We will do this in any case, since we want to end up with discrete spacetime, but this could be done later.

.. math::
   &{\bf x} = a {\bf n}, n_i = 0...N_s,\\
   &\int d^3x \to a^3 \sum_x\\
   &\partial_i\phi \to \frac 1a \left[ \phi({\bf x}+{\bf e}_ia) -\phi(x) \right] = \Delta_i\phi({\bf x})
   :label:

The Hamiltonian in discrete space is

.. math::
   &\hat H = a^3 \sum_x \left [ \frac 12 [\hat pi(x)]^2 + \frac 12 [\hat pi(x)]^2  \right ]
   :label:

Now let's consider the amplitude of the fields :math:`\phi_A` and :math:`\phi_B` in equilibrium at inverse temperature :math:`\tau = 1/T` and split the time interval into N small sections

.. math::
   &\left<\phi_B \left|e^{-\tau \hat H}\right|\phi_A \right> = 
    \left<\phi_B \left| \left(e^{-\tau/N \hat H}\right)^N\right|\phi_A \right>\\
   &= \left<\phi_B \left| \left(e^{-a_\tau \hat H}\right)^N\right|\phi_A \right>
   :label:

In order to evaluate this, we insert the identity operators :math:`1 = \int\left[\Pi_x\right]|\phi_x><\phi_x|` and :math:`\int\left[\Pi_x\right]|\pi_x><\pi_x|`.

.. math::
   &\left<\phi_B \left|e^{-\tau \hat H}\right|\phi_A \right> = \\
   &\int\left[\prod_{i=2}^N\prod_x d\phi_x\right]\left[\prod_{i=1}^N\prod_x d\pi_x \right]
   \left<\phi_B |\pi_N\right > \left<\pi_N |e^{-a_\tau \hat H}|\phi_N\right > \\
   &\left<\phi_N |\pi_{N-1}\right > \left<\pi_{N-1} |e^{-a_\tau \hat H}|\phi_{N-1}\right >\dots 
   \left<\phi_2 |\pi_{1}\right > \left<\pi_1 |e^{-a_\tau \hat H}|\phi_A\right >
   :label:

Now the matrix in each exponential is small, we can expand the first few two and conclude that

.. math::
   \left<\pi_n |e^{-a_\tau \hat H}|\phi_n\right > = 
   e^{-a^3 a_\tau \sum_x \left( \frac 12 \pi_x^2 + \frac 12 (\Delta_i\phi_n)^2 + V[\phi_i]\right )} \left< \pi_n |\phi_n \right> + O(a_\tau^2)
   :label:

and that

.. math::
   \left<\phi_{n+1} | \pi_n\right >\left<\pi_n | \phi_n\right > &= 
   e^{ia^3 \sum_x \pi_n({\bf x})\phi_{n+1}({\bf x}) } e^{-ia^3 \sum_x \phi_n({\bf x})\pi_n({\bf x}) } \\
   &= e^{-ia^3 a_\tau \sum_x \pi_n({\bf x})\Delta_0\phi_n({\bf x}) }
   :label:

where :math:`\Delta_0 = \frac {1}{a_\tau} \left( \phi_{n+1} - \phi_n \right)`.

Repeating this we can write the entire expression in terms of the field values :math:`\phi_n` and :math:`\pi_n` in discrete time.
Further, the intergral over the field :math:`\pi_n({\bf x})` is gaussian and 
after performing that integral we are only left with :math:`\phi_n`.

.. math::
   &\int\left[\prod_x d\pi_n({\bf x}) \right]
   e^{a^3 a_\tau \sum_x \left(-\frac 12 \pi_n({\bf x})^2 + (i\Delta_0\phi_n({\bf x})  \pi_n({\bf x})^2 \right) } \\
   &= \left(\frac{2\pi}{a^2a_\tau}\right)^{N_S^3/2} 
   e^{-a^3 a_\tau \sum_x \left(\Delta_0\phi_n({\bf x})\right)^2 }
   :label:

Now, after running through the same logic for all other :math:`n`, we find the path integral


.. math::
   \left<\phi_B \left| e^{-\tau \hat H}\right |\phi_A \right> = \frac 1N  \int\left[\prod_x d\phi_n({\bf x}) \right] \phi_A(\tau_A) \phi_B(\tau_B) e^{ -S_E } 
   :label:

and


.. container:: note

   .. math::
       S_E = a^3 a_\tau \sum_x \left (\frac 12 \Delta_\mu \phi \right )^2 + V[\phi] 
       :label:


In the continuum limit we would have

.. math::
   S_E = \int d^4x \left (\frac 12 \partial_\mu \phi \right)^2 + V[\phi] 
   :label:

 
So the path integral formulation in thermal equilibrium is has the same form as a
Euclidean field theory in four dimensions.
Furthern we can now write the partition function as

.. math::
   Z &= \int d\phi \left<\phi\left| e^{ -\hat H / T } \right|\phi\right>\\
     &= \int \left[d\phi \right] e^{ - \frac 1T S_E(\phi)}
   :label:

In summary

+-------------------------+------------------------------------------------------------------------------------+
| Minkowsky               |   Euclidean                                                                        |
+-------------------------+------------------------------------------------------------------------------------+
|:math:`\mathcal L_M`     |  :math:`\mathcal L_E = - \mathcal L_M|_{x_i \to ix_0; \partial_0 \to i\partial_0}` |
+-------------------------+------------------------------------------------------------------------------------+
|:math:`g = (1,-1,-1,-1)` |  :math:`g = (1,1,1,1)`                                                             |
+-------------------------+------------------------------------------------------------------------------------+


**Correlation functions**

The correlation functions of a classical theory are related to the Green's functions of the quantum field theory.
Earlier we used the "transfer matrix" :math:`T`,

.. math::
   T_{\phi_{i+1},\phi_i} = \left < \phi_{i+1} \left | e^{-a_\tau \hat H}\right | \phi_i \right > 
   :label:
We can write the partition function using the transfer matrix as

.. math::
   Z = \int \left [ d\phi \right ] e^{S_E} = Tr \left( T^{N_\tau} \right)
   :label:
Now diagonalizing the transfer matrix :math:`T` and labeling the eigenvalues as :math:`\lambda_0, \lambda_1,\dots`,

.. math::
   Z = \sum_i \lambda_i^{N_\tau} = \lambda_0^{N_\tau} \left[ 1 + O((\lambda_1/\lambda_0)^{N_\tau}) + \dots \right]
   :label:
Note that the eigenvalues :math:`\lambda_i` are equal to :math:`exp(E_i)`, where :math:`E_i` are the eigenvalues of :math:`\hat H`.
So :math:`\lambda_0` corresponds to the lowest energy state, or the vacuum :math:`|0>`.

The two point function (with :math:`i>j`) in the path integral and operator representations is

.. math::
   <\phi_i \phi_j> = \frac 1Z \int\left[ d\phi \right] \phi_i \phi_j e^{-S_E} 
   = \frac 1Z Tr\left( T^{N_\tau-(i-j)} \hat\phi T^{i-j} \hat\phi  \right) 
   :label:
Introducing a time dependent operator

.. math::
   \hat \phi(\tau) = e^{\tau\hat H} \hat\phi e^{-\tau\hat H}
   :label:
In the limit :math:`N_\tau\to\infty` (and because :math:`a_\tau(i-j) = \tau_i-\tau_j`),

.. math::
   <\phi_i \phi_j> = \left< 0 \left| \hat\phi \left(\frac{T}{\lambda_0}\right)^{i-j} \hat\phi \right| 0 \right>
   = \left< 0 \left| \hat\phi(\tau_i) \hat\phi(\tau_j) \right| 0 \right>
   :label:

Finally, if including also negative time separation :math:`i-j`, we have 

.. math::
   <\phi_i \phi_j> = \left< 0 \left| \mathcal T \left[ \hat\phi(\tau_i) \hat\phi(\tau_j) \right] \right| 0 \right>,
   :label:
where $\mathal T$ is the time ordering operator.


**Greens Function and the Mass Spectrum**

Any greens function for operator :math:`\Gamma` can be expanded in terms of energy states
(eigenstates of the hamiltonian)

.. math::
   \left< 0 \left| \Gamma(\tau) \Gamma^\dagger(0) \right| 0 \right>
   &= \frac 1Z \int\left[ d\phi \right ] \Gamma(\tau) \Gamma^\dagger(0) e^{-S}\\
   &= \left< 0 \left| e^{\hat H \tau} \Gamma(0) e^{-\hat H \tau} \Gamma(0) \right| 0 \right>\\
   &= \sum_n  \left< 0 \left| \Gamma(0) \right| E_n \right>  e^{-E_n \tau} \left< E_n \left | \Gamma(0) \right| 0 \right>\\
   &= \sum_n e^{-E_n\tau} \left| \left< 0 \left| \Gamma(0) \right| E_n \right> \right|^2
   :label:
At long enough distances we find

.. math::
   \left< 0 \left| \Gamma(\tau) \Gamma^\dagger(0) \right| 0 \right> 
   \to e^{-E_0\tau} \left| \left< 0 \left| \Gamma(0) \right| E_0 \right> \right|^2
   \textrm{ when } \tau \to \infty
   :label:
Here the state :math:`E_0` is the lowest energy eigenstate that the operator :math:`\Gamma(0)`
constructs out of the vacuum state and therefore the eigenvalue :math:`E_0` is the mass of a
state with the same quantum numbers as :math:`\Gamma(0)`.

This relation allows us to measure the masses of propagating composite states.
It is also useful for calculating certain more complicated observables, such as scattering lengths.



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
