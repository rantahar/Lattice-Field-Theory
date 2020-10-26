
**********************
Lattice Field Theories
**********************

.. toctree::
   :maxdepth: 2

   index
   



Motivation
==================

These lecture notes give an introduction to lattice field theory, a
powerful framework for solving quantum field theories from first
principles.
We approach the topic mainly from the point of view of QCD, going
through the building blocks necessary to simulate a model with fermions
with a gauge interaction.




Learning Objectives
==================

The course has two main objectives: to learn enough about lattice methods to put them into practical use and to become familiar with common methods used in studies of Standard Model physics.

After successfully completing the course, the student

-   can implement a Monte Carlo simulation of a quantum field theory in
    discrete space

-   can analyse the output of the calculation and describe its
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

.. raw:: html

   <img src="_static/spin_random.svg">


.. raw:: latex

   \includegraphics[width=0.6\linewidth]{spin_random.eps}



Each atom has a spin, which creates a small magnetic field. When the
spins point to different random directions (as above), they cancel each
other and the iron is not magnetized. But if for some reason they point
to the same direction, the small magnetic fields add up into a
macroscopic value.

.. raw:: html

   <img src="_static/spin_ordered.svg">

.. raw:: latex

   \includegraphics[width=0.6\linewidth]{spin_ordered.eps}

Why would this happen?

A spin pointing against a magnetic field has a slightly higher energy
than a spin pointing to the same direction as the field. So atoms close
to each other would prefer to align. At zero temperature, they would
eventually all point to the same direction and create a magnet. At higher
temperatures, thermal fluctuations will occasionally cause a spin to flip.

Let's only consider the spins closest to each other (nearest neighbors.)
The energy of two spins is

.. math::
   E=-j\mathbf{s}_{1}\cdot\mathbf{s}_{2}
   :label: 

and the energy of the whole system is

.. math::
   E=-J\sum_{<ij>}\mathbf{s}_{i}\mathbf{s}_{j}.
   :label:
The sum over :math:`<ij>` here counts pairs of neighbors. If the spins are
coupled to a magnetic field :math:`\mathbf{H}`, with the magnetic momentum
:math:`\gamma`, the energy is

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
     & =\int[\prod_{i}ds_{i}]e^{-\frac{J}{kT}\sum_{<ij>}\mathbf{s}_{i}\mathbf{s}_{j}+\frac{\gamma}{kT}\mathbf{H}\cdot\sum_{i}\mathbf{s}_{i}}
   :label:

The thermal expectation value of an observable :math:`O` is then

.. math::
   <O> = \frac 1Z \int[\prod_{i}ds_{i}] \, O(s) \, e^{-\frac{1}{kT}E(s)}
   :label:
 
At high temperatures the spins become essentially random and the
magnetization disappears.




The Ising Model
-----------------

The Ising model is a further simplification of the above. All the spins
are either :math:`+1` or :math:`-1`. The partition function then is

.. math::
   Z =\sum_{s_{i}=\pm1}e^{-\beta\sum_{<ij>}s_{i}s_{j}+h\cdot\sum_{i}s_{i}}.
   :label:
Here we use dimensionless couplings, :math:`\beta=\frac{J}{kT}` and
:math:`h=\frac{\gamma H}{kT}`.

The Ising model has been solved in 1 and 2 dimensions (Onsager 1944, Yang 1952).


**Simulation Algorithms**

The integral in the partition function has a very high dimension and cannot
practically be performed directly.
We introduce the Heat Bath algorithm here and will cover other Monte Carlo
methods in later chapters.
Monte Carlo methods perform very well when calculating high dimensional
integrals.

In the Ising model, the integral reduces to a sum over all possible configurations
of the spins. At constant temperature (in a heat bath), the probability of finding
the model in a given state should be equal to the Boltzmann weight.

.. math::
   P(s) = \frac{1}{Z} e^{-\beta E(s)}
   :label:

We use a model known as a Markov Chain to draw configurations from this distribution.
Starting from any given configuration :math:`s_0`, we build a new configuration
:math:`s_1` and accept it with an appropriate probability :math:`W_f(s_0 \to s_1)`.
The probability needs to be chosen correctly so that we do not move out of the
Boltzmann distribution.

The update probability needs to satisfy two conditions:

**1. Detailed Balance**

.. math::
   \frac{W_f(s_0\to s_1)}{W_f(s_1 \to s_0)} = \frac{P(s_1)}{P(s_0)} = e^{-\beta [ E(s_1) - E(s_0) ]}
   :label:

If the first configuration is drawn with the correct probability, so will be
the second one. Detailed balance is in fact a bit more stringent a requirement
than is necessary, but it's usually easiest to implement an algorithm that
satisfies it.

**2. Ergodicity**

It must be possible to reach any possible configuration from any other possible
configuration in a finite number of updates. Otherwise we might get stuck in 
a finite region of the space of configurations.

With these two conditions, as long as the original configuration is drawn from
the correct distribution, the next one will also be.
And since all configurations have a non-zero weight, we can actually start from
any configuration.

But how do we construct the update? The Ising model has been studied for a long
time and there are many options.
One option is to update one spin at a time, flipping each spin with the appropriate
probability to satisfy detailed balance. Here are two common options for
the probability a spin:

**1. Heat Bath Algorithm**:

In the heat bath method, we choose a random new spin based on its Boltzmann weight
normalized by the sum of all choices (in the Ising model just :math:`\pm 1`).
The probability of choosing a new state :math:`s_1` from possible states :math:`s`
is

.. math::
   W_f(s_0\to s_1) = \frac{ P(s_1) }{ \sum_s P(s) }
   :label:

Notice that the probability does not depend on the previous state :math:`s_0`.
So the probability of choosing a positive spin is

.. math::
   P_+ = \frac{ P(1) }{ P(1) + P(-1) }
   :label:
and for a negative spin

.. math::
   P_- = \frac{ P(-1) }{ P(1) + P(-1) }
   :label:


**2. Metropolis Algorithm** (Metropolis-Hastings Algorithm):

In the Metropolis method we suggest a random change to the current state
of the model. Given the initial state :math:`s_0` the probability of choosing
the new state :math:`s_1` is 

.. math::
   W_f(s_0\to s_1) = min\left( 1, e^{- \beta [ E(s_1) - E(s_0) ]} \right )
   :label:

So in the Metropolis method the probability of changing the spin depends
on the initial state. If the energy decreases, the update is always accepted.
If it increases, the probability is the ratio of Boltzmann weights of the two
options. You can check that the update follows detailed balance in both cases.

Note that if there were more than one possible spin, we would need to choose between
them using some update probability. This probability would need to be chosen so that
detailed balance is still preserved. More on this later, when we update models with
continuous parameters.


.. container:: note

   **Example**

   Let's implement the Ising model. This is in python, but it's hopefully readable
   if you know another language. The algorithm is the important part. I recommend that
   you write the code yourself, and don't just copy and paste.
   
   First will need numpy

   .. code-block:: python

      import numpy as np
   
   Next set up some parameters

   .. code-block:: python

      lattice_size = 16
      temperature = 4.0
      number_of_updates = 1000

   Create a numpy array that contains a spin on each lattice site

   .. code-block:: python

      spin = np.ones([lattice_size,lattice_size], dtype=int)

   and run updates in a loop, updating one random site at a time

   .. code-block:: python

      for i in range(number_of_updates):
         x = int( lattice_size*np.random.random() )
         y = int( lattice_size*np.random.random() )

   Now we will randomly change the spin so that the probability matches the Boltzmann
   distribution. First calculate the probability weights of each state

   .. code-block:: python

         energy_plus  = -(1) * ( spin[x][(y+1)%lattice_size] + spin[x][y-1] 
                               + spin[(x+1)%lattice_size][y] + spin[x-1][y] )
         energy_minus = -(-1) * ( spin[x][(y+1)%lattice_size] + spin[x][y-1] 
                                + spin[(x+1)%lattice_size][y] + spin[x-1][y] )
         
         P_plus  = np.exp( -energy_plus/temperature )
         P_minus = np.exp( -energy_minus/temperature )
   
   Notice the module lattice size (%lattice_size) in the neighbor index. If the current site
   is at the lattice boundary, this will point to the other side. We don't need this in the
   negative direction since python does it automatically.

   Now calculate the heat bath probability of choosing positive spin and we choose the spin
   according to this probability

   .. code-block:: python

      probability_plus = P_plus / (P_plus + P_minus)

      if np.random.random() < probability_plus:
         spin[x][y] = 1
      else:
         spin[x][y] = -1

   This works in principle, but the program is a bit boring since it doesn't measure anything.
   Let's print out the energy after each update.

   .. code-block:: python

      energy = 0
      for x in range(lattice_size):
         for y in range(lattice_size):
            energy += - spin[x][y] * spin[x][(y+1)%lattice_size]
            energy += - spin[x][y] * spin[(x+1)%lattice_size][y]
      
      print("The energy is {}".format(energy))
   
   
   **Exercise**
   
   1. Measure the magnetization as well
   2. Running the measurement between each update is really wasteful.
      Do multiple updates between measurements.
   3. Switch to the Metropolis Algorithm


   

Other Spin Models
------------------

Several more complicated spin models are studied. Some of the simpler ones include

**The Potts Model**

Similar to the Ising model, but there are :math:`N` possible spins. Neighboring spins have
lower energy if they are equal.

.. math::
   E = -\beta\sum_{<ij>} \delta(s_{i},s_{j}) + |h| \sum_{i} \delta(\hat h, s_{i})
   :label:


**The XY Spin Model**

Each spin is represented by an angle :math:`\alpha`, or a 2D vector :math:`s` of length 1.
The distance between spins is represented by the dot product, or equivalently the cosine
of the difference of the angles.

.. math::
   E &= -\beta\sum_{<ij>} \cos( \alpha_i - \alpha_j ) + |h| \sum_{i} \cos( \alpha_i - \alpha_h )\\
     &= -\beta\sum_{<ij>} s_{i} \cdot s_{j} + h \cdot \sum_{i} s_{i}
   :label:


**O(N) Spin Models**

Similarly, spins can be represented by N-dimensional vectors.

.. math::
   E = -\beta\sum_{<ij>} s_{i} \cdot s_{j} + h \cdot \sum_{i} s_{i}
   :label:




Observables
-----------

We can measure the amount of magnetization through the sum of the spins.
For an individual configuration


.. math::
   M=\frac{1}{V}\sum_{i}s_{i},
   :label:

where V is the number of points on the lattice, the volume. We get the thermal average by integrating over the Boltzmann distribution:


.. math::
   <M> =\frac{1}{V} \frac 1Z \int[\prod_{i}ds_{i}]  e^{-\frac{1}{kT}E(s)} ( \sum_{i}s_{i}) 
   :label:

This can also be expressed as a derivative of the partition function with respect to the external field :math:`h` 

.. math::
   <M> = \frac{1}{V} \frac{\partial}{\partial h} \log(Z).
   :label:
So the field :math:`h` functions as a source for the magnetization.

Similarly the energy is 

.. math::
   <E> &= \frac 1Z \int[\prod_{i}ds_{i}] \, E(s) \, e^{-\beta E(s)} \\
   &= -\frac{\partial}{\partial \beta} \log(Z)
   :label:

Other interesting observables include
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
   \chi_M & = \frac{\partial}{\partial h} <M> = \frac 1V \frac{\partial^2}{\partial^2 h} \log(Z) \\
   &= V\left( <M^2> - <M>^2 \right)
   :label:

- Correlation functions

.. math::
   &C(\mathbf{x}-\mathbf{y}) = <s_{\mathbf{x}} s_{\mathbf{y}}> - <s_{\mathbf{x}}><s_{\mathbf{y}}>, \\
   &\lim_{|\mathbf{x} - \mathbf{y}|\to\infty} C(\mathbf{x}-\mathbf{y}) = e^{-|\mathbf{x} - \mathbf{y}|/\xi},
   :label:
where :math:`\xi` is the correlation length.

Deriving this from the partition function requires introducing an :math:`\mathbf{x}`-dependent source :math:`h_\mathbf{x}`

.. math::
   Z =\sum_{s_{i}=\pm1}e^{\beta\sum_{<ij>}s_{i}s_{j}+\sum_{i} h_i s_{i}}.
   :label:

.. math::
   C(\mathbf{x}-\mathbf{y}) = \partial_\mathbf{x} \partial_\mathbf{y}
     \left . \log(Z) \right |_{h=0}
   :label:



Transfer matrices (Example of an exact solution)
-----------

Consider the 1D Ising model:

.. raw:: html

   <img src="_static/1Dlat.svg">


.. raw:: latex

   \includegraphics[width=0.6\linewidth]{1Dlat.eps}

and assume periodic boundary conditions

.. math::
   s_{x+L}=s_x
   :label:

First we'll write the energy in a symmetric way between the neighboring sites

.. math::
   E &= \beta\sum_{x=1}^L s_x s_{x+1} +h\sum_{x=1}^L s_x\\
     &=\sum_{x=1}^L \left( \beta s_x s_{x+1} + \frac 12 h (s_x + s_{x+1}) \right)
   :label:

We'll define the :math:`2\times2` transfer matrix

.. math::
   T_{s,s'} = e^{\beta s s' + \frac 12 h(s+s')}.
   :label:
Now the partition function can be written as

.. math::
   Z &= \sum_{\{s_x\}} T_{s_1,s_2} T_{s_2,s_3} \dots T_{s_{L-1},s_L} T_{s_L,s_1}\\
     &= \sum_{s_1} \left( T^L \right)_{s_1,s_1}\\
     &= Tr (T^L)
   :label:

The transfer matrix describes the contribution to the energy between two connected
spins, :math:`s` and :math:`s'`.

Writing the matrix explicitly,

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
   \log(Z) &= \log \left( Tr \left (\begin{matrix}
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
   \log(Z) = \log(\lambda_+) = L\log \left( \cosh(h) + \sqrt{1 + \sinh^2(h)+e^{-4\beta}} \right ) + L\beta
   :label:

From here we can calculate the magnetization as a function of :math:`h`

.. math::
   <M> &= \frac 1L \frac \partial {\partial h} \log(Z)
   = \frac{\sinh(h) + \frac{\cosh(h) \sinh(h)}{\sqrt{\sinh^2(h)+e^{-4\beta}} }}{\cosh(h)+\sqrt{\sinh^2(h)+e^{-4\beta}} }\\
   &= \frac{\sinh(h)}{\sqrt{\sinh^2(h)+e^{-4\beta}} }
   :label:

So at :math:`h=0` the magnetization is zero, which is what we expect. At large :math:`\beta`, small temperature, it approaches one, which is also expected. Here is a sketch of its behavior in general:

.. raw:: html

   <img src="_static/ising_solutions.svg">


.. raw:: latex

   \includegraphics[width=0.6\linewidth]{ising_solutions.eps}





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

- **Potts Model:**


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

- If :math:`h\neq 0` the symmetry is explicitly broken and :math:`<M>\neq 0`

Non-trivial symmetry breaking happens in the thermodynamic limit, :math:`V\to 0`. 
The symmetry is spontaneously broken if

.. math::
   \lim_{h\to 0} \left[ \lim_{V\to\infty} <M> \right ] \neq 0
   :label:
The order of the limits is important here. If the limit :math:`h\to0` is taken too quickly the magnetization will approach :math:`0`.

- The 1D Ising model the symmetry is *not* spontaneously broken. 

.. math::
   \lim_{N\to\infty} <M> =  \frac{\sinh(h)}{\sqrt{\sinh^2 + e^{-4\beta}}}
   :label:

- At :math:`D>1` the symmetry is broken at :math:`T>T_c`, or :math:`\beta < \beta_c`.


**Phase transitions**


In many models we have a broken symmetry at :math:`\beta > \beta_c` and a restored symmetry at :math:`\beta < \beta_c`. This means there is a phase transition at :math:`\beta=\beta_c`

 - **First order**
 One or more of the first derivatives of the free energy
 
.. math::
   F=-\frac{1}{\beta}\log(Z)
   :label:
is discontinuous:

.. math::
   <E> = \frac{\partial}{\partial\beta} \beta F
   :label:

.. math::
   <M> = -\frac 1V \frac{\partial}{\partial h} \beta F
   :label:



.. raw:: html

   <img src="_static/transition_1.svg">


.. raw:: latex

   \includegraphics[width=0.6\linewidth]{transition_1.eps}


The jump in energy is known as the latent heat

.. math::
   \frac 1V \Delta E = \lim_{\beta\to_-\beta_c}<E> - \lim_{\beta\to_+\beta_c}<E> = \frac{E_- - E_+}{V}
   :label:

How does this work on the lattice? The transition is not instantaneous,
but gets smoothed over a small span of temperatures.
The derivative of the energy in fact grows with the volume.

.. math::
   \chi = \frac 1V \left<(E-<E>)^2\right> 
   :label:
At :math:`\beta=\beta_c`, :math:`<E>\approx \frac 12 (E_+ + E_-)`, so

.. math::
   \chi \approx \frac 1V \frac{\Delta E^2}{4}
   = V \frac 14 \left( \frac{\Delta E}{V} \right)^2 \sim V
   :label:

In a first order transition the two phases can coexist, such as ice and water. The average energy density in this state is between the two phases.


 - **Second order**

No discontinuity in the first derivative of the free energy, 
but there is a discontinuity in the second derivative.
This is the case in spin models.
The derivative of magnetization

.. math::
   \frac{\partial}{\partial_h}<M> \neq 0
   :label:

is discontinuous.

.. raw:: html

   <img src="_static/transition_2.svg">


.. raw:: latex

   \includegraphics[width=0.6\linewidth]{transition_2.eps}




Critical Phenomena:
""""""""""""""""""""""

The correlation length :math:`\xi` diverges exponentially at :math:`\beta_c`.
Structures scale to all distances:

Writing :math:`\tau = \beta-\beta_c`:

.. math::
   \chi &\sim |\tau |^{-\alpha} \\
   \chi_M &\sim |\tau |^{-\gamma} \\
   \xi &\sim |\tau |^{-\nu} 
   :label:


.. math::
   <M> &= 0 \textrm{ at } \beta \leq \beta_c \textrm{ and }\\
   <M> &\sim |\tau|^\delta \textrm{ at } \beta > \beta_c
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
when :math:`\beta > \beta_c` :math:`h\neq 0`, if we change :math:`h` continuously across :math:`0`.

-  **Crossover**
  
Crossover is a term for a transition, where no symmetry is broken and / or 
there is no discontinuity in the derivative of the free energy.
The transition is continuous and there is no well defined critical
temperature, but the two phases are nevertheless distinct.



Field theories
============================

Quantum Field Theories
-----------------------

Quantum field theory is a huge field and introducing it properly is outside the scope of this
course. In short, quantum field theory unifies the *special* theory of relativity and quantum mechanics.
This is not an easy task for point particles, but
it turns out you the quantum mechanical version of relativistic electromagnetism works.
Quantum field theory builds on this observation and uses field theories to describe other particles
as well.

Let's consider a scalar field theory is Minkowsky spacetime (the field :math:`\phi` could also represent a more complicated set of fields). Given the action

.. math::
   S = \int d^3 dt \mathcal L(\phi,\partial_t \phi) = \int d^3 dt \left [ \frac 12 \partial_\mu \phi \partial^\mu\phi - V(\phi) \right ]
   :label:
The classical equations of motion are found by minimizing the action, which leads to the 
Euler-Lagrange equation

.. math::
   \partial_\mu \frac{\delta L}{\delta(\partial_\mu \phi)} - \frac{\delta L}{\delta \phi} = 0.
   :label:
In this case we find the Klein-Gordon equation

.. math::
   \partial_\mu \partial^\mu \phi = -\frac{dV(\phi)}{d\phi}.
   :label:

The classical Hamiltonian is obtained by a Legendre transformation

.. math::
   \int d^3xdt H = \int d^3xdt\left [  \pi\dot\phi-\mathcal L \right ],
   :label:

where :math:`\pi = \delta \mathcal L/\delta\dot\phi` is the canonical momentum, and

.. math::
   H(\phi,\pi) = \pi^2 +\frac 12 (\partial_i\phi)^2 + V(\phi).
   :label:

In quantum field theory, we consider the Hilbert space of states :math:`\ket{\phi,\pi}`.
These evolve according to the Schrödinger equation

.. math::
   i\hbar\partial_t \ket{\phi,\pi} = \hat H \ket{\phi,\pi}.
   :label:

Here the Hamilton operator is

.. math::
   \hat H \ket{\phi,\pi} = H(\phi,\pi) \ket{\phi,\pi}.
   :label:
The expectation value of a measurable that depends on the fields :math:`\phi` and :math:`\pi`
is

.. math::
   \Braketmid{\phi,\pi}{O(\phi,\pi)}{\phi,\pi}
   :label:

and

.. math::
   &\int \left[\prod_x d\phi({\bf x})\right] \ket{\phi}\bra{\phi} = 1\\
   &\int \left[\prod_x \frac {d\pi({\bf x})}{2\pi} \right] \ket{\pi}\bra{\pi} = 1\\
   &\Braket{\pi}{\phi} = e^{i\int d^3xdt \pi({\bf x})\phi({\bf x})}
   :label:

We also define field operators

.. math::
   &\hat \phi({\bf x}) \ket{\phi} = \phi({\bf x}) \ket{\phi}\\
   &\hat \pi({\bf x}) \ket{\pi} = \pi({\bf x}) \ket{\pi}\\
   &[\hat\phi({\bf x}),\hat\phi({\bf x}')] = -i\delta^3({\bf x} - {\bf x}') \\
   :label:

We can formally solve the Schrödinger equation to find the time evolution operator

.. math::
   U(t)=e^{\frac{i}{\hbar}\hat H t}.
   :label:
In this representation, we can define a partition function analogically to classical statistics,

.. math::
   Z = \textrm{Tr} e^{\frac{i}{\hbar}\hat H t} = \int d\phi \left<\phi\left| e^{\frac{i}{\hbar} \hat H t} \right|\phi\right>.
   :label:
From here we could derive the Feynman path integral representation by 
evolving the a field configuration :math:`\bra{\phi,\pi}` by small time steps :math:`\delta t` and 
taking the limit :math:`\delta t\to 0`. Here we will skip the derivation and simply introduce the result.
Then we will follow essentially the same steps in field theory at thermal equilibrium.


The Path Integral Representation
-----------------------------------

First, let us quickly introduce the path integral representation in Minkowsky space. We will not
derive this, since the derivation is essentially the same in thermal field theory and we will do
it there.
The expectation value of an observable is

.. math::
   &<O> = \frac{1}{Z} \int \left [ \prod_x d\phi(x) \right ] O(\phi) e^{\frac{i}{\hbar} S(\phi)} \\
   & \textrm{ where } Z= \int \left [ \prod_x d\phi(x) \right ]e^{\frac{i}{\hbar} S(\phi)}\\
   & \textrm{ and } S=\int d^4x \mathcal L (\phi) 
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

There are no state vectors or operators in the path integral representation. Instead, the variables closely
resemble classical fields. The difference is the integral over all possible field values, with a weight
defined by the action. 

Now, since the fields are defined at all space-time locations, the integral measure

.. math::
   \prod_x d\phi(x) 
   :label:
is not well defined and needs to be regularized. This is a general problem with functional integrals.
Lattice field theory is a conceptually simple renormalization method:
we divide the volume into a lattice of discrete points :math:`x\in aZ^4` and study a system with
a finite volume :math:`V`. 
Since the integral is now finite, we can in principle calculate is directly (brute force, with supercomputers,)
and get fully non-perturbative results.

The full theory is recovered by taking the infinite volume and continuum limit :math:`v\to \infty, a\to0`.
The order of the limits is important here, just like for the spin models.

In practice the dimensionality of the integral grows quickly when increasing the volume and decreasing the lattice spacing.
In most cases the integral can be calculated directly only for lattice sizes that are practically useless.

Instead, we should use Monte Carlo methods.
The problem here is the complex, unimodular weight, :math:`\exp(iS)`.
Every configuration :math:`\{\phi\}` contributes with the same magnitude and result depends on cancellations between configurations.
However, this is (mostly) solved in the imaginary time formalism of thermal field theory.



Imaginary Time Path integral
----------------------------


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
   \hat H = a^3 \sum_x \left [ \hat \pi^2(x) + \frac 12 [\Delta_i \hat \phi(x)]^2 +V(\phi) \right ]
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

Since the matrix in each exponential is small, we can expand the first few two and conclude that

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
Further, the integral over the field :math:`\pi_n({\bf x})` is gaussian and 
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

 
Finite Temperature
------------------

So the path integral formulation in thermal equilibrium has the same form as a
Euclidean field theory in four dimensions.
Further we can now write the partition function as

.. math::
   Z &= \int d\phi \left<\phi\left| e^{ -\hat H / T } \right|\phi\right>\\
     &= \int \left[d\phi \right] e^{ - \frac 1T S_E(\phi)}
   :label:

The quantum statistical model in 3 dimensions corresponds to a classical statistical model
in 4 dimensions with periodic boundary conditions.


In summary

+-------------------------+------------------------------------------------------------------------------------+
| Minkowsky               |   Euclidean                                                                        |
+-------------------------+------------------------------------------------------------------------------------+
|:math:`\mathcal L_M`     |  :math:`\mathcal L_E = - \mathcal L_M|_{x_i \to ix_0; \partial_0 \to i\partial_0}` |
+-------------------------+------------------------------------------------------------------------------------+
|:math:`g = (1,-1,-1,-1)` |  :math:`g = (1,1,1,1)`                                                             |
+-------------------------+------------------------------------------------------------------------------------+


**Correlation functions**

The correlation functions of a classical theory are related to the Greens functions of the quantum field theory.
Earlier we used the "transfer matrix" :math:`T`,

.. math::
   T_{\phi_{i+1},\phi_i} = \Braketmid{ \phi_{i+1} }{ e^{-a_\tau \hat H} }{ \phi_i } 
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
   <\phi_i \phi_j> = \Braketmid{ 0 }{ \hat\phi \left(\frac{T}{\lambda_0}\right)^{i-j} \hat\phi }{ 0 }
   = \Braketmid{ 0 }{ \hat\phi(\tau_i) \hat\phi(\tau_j) }{ 0 }
   :label:

Finally, if including also negative time separation :math:`i-j`, we have 

.. math::
   <\phi_i \phi_j> =\Braketmid{ 0 }{ \mathcal T \left[ \hat\phi(\tau_i) \hat\phi(\tau_j) \right] }{ 0 },
   :label:
where :math:`\mathcal T` is the time ordering operator.


**Greens Function and the Mass Spectrum**

Any Greens function for an operator :math:`\Gamma` can be expanded in terms of energy states
(eigenstates of the hamiltonian)

.. math::
   \Braketmid{ 0 }{ \Gamma(\tau) \Gamma^\dagger(0) }{ 0 }
   &= \frac 1Z \int\left[ d\phi \right ] \Gamma(\tau) \Gamma^\dagger(0) e^{-S}\\
   &= \Braketmid{ 0 }{ e^{\hat H \tau} \Gamma(0) e^{-\hat H \tau} \Gamma(0) }{ 0 }\\
   &= \sum_n  \Braketmid{ 0 }{ \Gamma(0) }{ E_n }  e^{-E_n \tau}  \Braketmid{ E_n }{ \Gamma(0) }{ 0 }\\
   &= \sum_n e^{-E_n\tau} \left| \Braketmid{ 0 }{ \Gamma(0) }{ E_n } \right|^2
   :label:

At long enough distances we find

.. math::
   \Braketmid{ 0 }{ \Gamma(\tau) \Gamma^\dagger(0) }{ 0 } 
   \to e^{-E_0\tau} \left| \Braketmid{ 0 }{ \Gamma(0) }{ E_0 } \right|^2,
   \textrm{ when } \tau \to \infty
   :label:

Here the state :math:`E_0` is the lowest energy eigenstate that the operator :math:`\Gamma(0)`
constructs out of the vacuum state and therefore the eigenvalue :math:`E_0` is the mass of a
state with the same quantum numbers as :math:`\Gamma(0)`.

This relation allows us to measure the masses of propagating composite states.
It is also useful for calculating certain more complicated observables, such as scattering lengths.


**Lattice Terminology**

We do computation on a lattice with a finite number of points, so a finite volume and finite lattice spacing.
In order to obtain results in a continuum model, we need to take two limits in order:
 - :math:`V \to \infty`: the *thermodynamic* limit
 - :math:`a \to 0`: the *continuum* limit

 1. Run simulations at fixed :math:`a` and several :math:`V`. Extrapolate to :math:`V\to\infty`
 2. Do the previous with different values of :math:`a` and extrapolate to :math:`a\to 0`

Often it is also easier to run simulations with unphysical parameters, such as quark mass in QCD
and take the limit :math:`m_q \to m_{q,phys}`

At :math:`T=0`

 1) :math:`V\to \infty`

    :math:`N_\tau,N_s\to\infty`, :math:`a` constant
 2) continuum:
    :math:`a\to 0`

At :math:`T>0`

 1) :math:`V\to \infty`

    :math:`N_s\to\infty`, :math:`a` constant, :math:`N_\tau a` constant
 2) continuum:
    :math:`a\to 0`, :math:`\frac 1T = aN_\tau` constant




Scalar Fields
==============

The action
-------------

The continuum action of a free scalar field :math:`\phi` in Euclidean space is

.. math::
   S = \int dx \frac 12 \sum_\mu \phi(x) \partial_\mu \partial_\mu \phi(x) + \frac 12 m^2\phi(x)^2
   :label:

On a lattice we can define the d'Alembert operator as

.. math::
   \Box_{x,y} = \sum_\mu \frac {1}{a^2} \left( 2\delta_{x,y} - \delta_{x+\hat\mu,y}  - \delta_{x-\hat\mu,y}\right)
   :label:

so that

.. math::
   \sum_y \Box_{x,y} \phi_y = -\Delta^2\phi = \sum_\mu \frac {1}{a^2} \left( 2\phi_x - \phi_{x-\hat\mu} - \phi_{x+\hat\mu} \right)
   :label:
where :math:`\hat\mu` is a basis vector of length :math:`a` and :math:`x` and :math:`y` are vectors,
:math:`x_\mu = an_\mu`
Using these definitions, we can write the lattice action as

.. math::
   S =  a^d \sum_x\left[ \frac 12 \sum_y \phi_x \Box_{x,y} \phi_y + \frac 12 m^2\phi_x^2 \right]
   :label:

The partition function is Gaussian, :math:`S=\frac12 \sum_{x,y} \phi_xM_{x,y}\phi_y`, and we can actually
do the integral

.. math::
   Z = \int \left[d\phi\right] e^{-S} = \left( \det \frac {M}{2\pi}  \right)^{-1/2}
   :label:

While this looks like it should reproduce the continuum action when :math:`a\to 0`, this is
actually a subtle issue. In this case things will work out, but once we get to fermions, we will see
how this can go wrong. In any case, we should check that our action produces the correct model in 
continuum.

Scalar Field Theories
----------------------

Most classes in field theories will introduce the :math:`\phi^4`-theory:

.. math::
   L_4(\phi) &= \sum_x \left [ \frac 12 \sum_y \phi_x \Box_{x,y} \phi_y + \frac 12 m^2\phi_x^2 
       + \frac{\lambda}{4!} \phi_x^4 \right ] \\
   Z &= \int [d\phi] e^{-L_4(\phi)}
   :label:


The Higgs field is a complex scalar field, :math:`\phi_x` is a complex number.
The partition function, excluding interactions with other standard
model fields is

.. math::
   L_H(\phi) &= \sum_x \left [ \frac 12 \sum_y \phi^*_x \Box_{x,y} \phi_y - \mu^2\phi_x^2 
       + \lambda \phi_x^4 \right ] \\
   Z &= \int [d\phi] e^{-L_H(\phi)}
   :label:

This is similar to the :math:`\phi^4`-theory, but the *mass* term is negative. This leads
to a potential well at zero temperature. As a results, the Higgs model has a symmetry
breaking transition at a finite temperature.



Updating Scalar Fields
-----------------------

Scalar fields are continuous, represented by floating point numbers.
The most efficient update method depends on the theory, but the metropolis update
gives a very general method:

1. Choose a random field on a random site

2. Update the field by adding a random value :math:`\Delta`. Gaussian distribution usually 
   works well. The parameter C is tunable.

.. math::
   \phi_x \to \phi + C \Delta
   :label:

3. accept or reject the change according to the metropolis probability.




Fourier Transforms
---------------------

The discrete Fourier transform is

.. math::
   \tilde f(k) = \sum_x a^d e^{-ikx} f(x)
   :label:

On a lattice with finite lattice spacing :math:`a`, :math:`x=an`, the momentum is periodic

.. math::
   \tilde f(k+2\pi m/a) = \tilde f(k)
   :label:

and we can restrict it to the Brillouin zone, :math:`-\pi < ak_\mu \leq \pi`

In infinite volume the inverse transform is

.. math::
   f(x) = \int_{-\pi/a}^{\pi/a}\frac{d^dk}{(2\pi)^d}e^{ikx}\tilde f(k)
   :label:

It is often more convenient to use dimensionless units, where :math:`x_\mu \in Z,`
:math:`-\pi < k_\mu \neq \pi`.

If the lattice is finite, the inverse transform depends on the boundary conditions.
With periodic boundaries, :math:`x_\mu +aN = x\mu`, the allowed momenta are 
:math:`ak_\mu = \frac{2\pi}{N} n_\mu -\pi`, with :math:`0<n_\mu \leq N`.
The inverse transform is a sum over these momenta:

.. math::
  f(x) &= \sum_k \frac {1}{(aN)^d} e^{ikx}\tilde f(k), \\
  k_\mu &=\frac{2\pi}{aN} n_\mu -\pi
  :label:

This approaches the integral form when :math:`N\to \infty`


Lattice Propagator
---------------------

The scalar field propagator :math:`G(x,y)` is the inverse (Greens function) of the operator
:math:`a^{-d}M = (\Box + m^2)`,

.. math::
   \sum_y a^d ( \Box_{x,y} + \delta_{x,y} m^2 ) G(y,z) = \delta_{x,z}
   :label:

Calculating Greens functions is often simplest by taking a Fourier transform:

.. math::
   \left [ \sum_\mu \frac {2}{a^2} (1-cos(k_\mu a)) + m^2 \right ] \tilde G(k) = 1
   :label:

So the propagator is

.. math::
   \tilde G(k) = \frac{1}{\hat k^2 + m^2},
   :label:

where :math:`\hat k` is the lattice momentum

.. math::
   \hat k^2 = \sum_\mu \hat k_\mu^2
   = \sum_\mu \left[ \frac 2 a \sin\left( \frac{ak_\mu}{2} \right) \right]^2
   :label:

In the continuum limit we take :math:`a\to 0` and

.. math::
   \hat k^2 &= k^2 + O(a)\\
   \tilde G(k) &= \frac{1}{k^2+m^2} + O(a)
   :label:

In this case everything works well. The propagator behaves correctly in
the continuum and describes a single particle.


**Poles of the Propagator**

In Minkowsky spacetime the propagator has a pole at
:math:`k_0^2 = k_ik^i + m^2` and this defines the dispersion
relation of a free particle.
In Euclidean time there is no pole since the square in the denominator
is always positive.

We can recover the pole structure using a Wick rotation back into
Minkowski space: :math:`k_0^M \to ik_0^E`. This works since the propagator
has no poles in the quarter of complex space between :math:`k_0^E` and 
:math:`k_0^M`.

.. math::
   \left[ \frac 2 a \sin\left( \frac{iak_0}{2} \right) \right]^2
   + \sum_i \hat k_i^2 + m^2
    = \left[ \frac 2 a \sinh\left( \frac{ak_0}{2} \right) \right]^2
   + \sum_i \hat k_i^2 + m^2 = 0
   :label:

So

.. math::
   ak_0 = 2 \sinh^{-1} \sqrt{ \sum_i \sin^2\frac {ak_i}{2} + \frac{(am)^2}{4} }
   :label:


Here is a comparison of the lattice and continuum propagators at :math:`am=0.1`

.. raw:: html

   <img src="_static/lattice_continuum_scalar_propagator.svg">


.. raw:: latex

   \includegraphics[width=0.6\linewidth]{lattice_continuum_scalar_propagator.eps}
   


**N-point Functions**

The Greens functions can be generated using a source :math:`J`

.. math::
   S(J) = \sum_x a^d \left[ \frac 12 \phi_x (\Box + m^2) \phi_x - J_x \phi_x \right]
   :label:

with

.. math::
   Z(J) = \int [d\phi] e^{-S(J)} = Z(0) e^{\sum_{x,y} a^{2d} \frac 12 J_x G(x,y) J_y}
   :label:

This defines N-point functions

.. math::
   \ev{\phi_x,\dots,\phi_y} = \left . \frac{1}{Z(0)}
   \frac{\partial}{\partial J_x} \cdots \frac{\partial}{\partial J_y}
   Z(J) \right |_{J=0} 
   :label:






Monte Carlo Methods
====================

Now let's take a brief look at Markov Chain Monte Carlo algorithms.
We already used these in the Ising model example, but we need to go
over a bit more detail in order to analyze our results properly.

**Dartboard Example**

The dartboard example clarifies the general idea of the Monte Carlo method.
Monte Carlo is a general method for numerically evaluating an integral.
Say we need to calculate the area of a circle with unit radius.
The area is

.. math::
   V = \pi r^2 = \pi
   :label:

But assume we don't know this, but we do know the area of a square.
We can evaluate the are of any shape within the square by randomly throwing
darts into the square. The probability of a dart landing in the circle is

.. math::
   p(\textrm{hits inside circle}) = \frac{A_{circle}}{A_{square}} \approx
   \frac{\textrm{number of hits inside circle}}{\textrm{number of hits inside square}}
   :label:

.. math::
   \lim_{N\to\infty} \frac 1N \sum_{i=0}^N \delta({\bf x}_i \textrm{ in circle})
    = \frac{\pi}{4}
   :label:

This method can be used to evaluate the area of any irregular shape as
long as we can determine whether the dart has landed inside it.



Monte Carlo integration
-------------------------

Now let's take a look at a 1D integral

.. math::
   I = \int_a^b dx f(x)
   :label:

We could take this to describe a shape inside a square and 
draw random values on both x and y axis.
We would need to evaluate the function at x to determine whether
:math:`y<f(x)`.

.. math::
   I &\approx \frac{(b-a)(y_{max}-y_{min})}{N} \sum_{i=0}^N \delta(y_i < f(x_i))\\
   &\approx \frac{(b-a)(y_{max}-y_{min})}{N} \sum_{i=0}^N \frac{f(x_i)}{y_{max}-y_{min}}\\
   &\approx \frac{b-a}{N} \sum_{i=0}^N f(x_i)
   :label:

In the 1D case this this is not the most efficient method, but for large
dimensional integrals the error of this method scales favorably.
The other standard method for numerical integration is to split the domain
into equal pieces and sum them up, following a simple rule for each piece.
For example, if following the *trapezoidal rule*, the error scales as

.. math::
   \delta \sim \frac {1}{N^{2/D}}.
   :label:

In error of the Monte Carlo method is statistical and it follows from the
*central limit theorem* that

.. math::
   \delta \sim \frac {1}{\sqrt{N}}.
   :label:

We will not prove this here, but you can check it by calculating the
standard deviation in your own simulations.

.. container:: note
   
   **Exercise**
   
   1. Run a simulation of the Ising model and print out 
      a large number of measurements. Take the first 100, 200, 500
      and  1000 measurements and check this follows the
      inverse square root scaling.


So the Monte Carlo method becomes more efficient around :math:`D = 8`.
In a model with a single scalar field each site has a single spin, so
for a small :math:`4\times4\times4\times4` lattice we already have

.. math::
   D = 4\times4\times4\times4 = 256
   :label:


This result is true as long as the function and its square are 
integrable 

.. math::
   \int_0^1 f(x) = \textrm{finite}\\
   \int_0^1 f^2(x) = \textrm{finite}
   :label:

This is true in lattice simulations, but here is a counterexample:

.. math::
   I=\int_0^1 \frac {1}{\sqrt{x}}
   :label:

The mean and standard deviation from a Monte Carlo run are still
correct, but the error may not scale as inverse square root.



Importance Sampling
----------------------

The Monte Carlo Method can be made even more efficient using importance sampling.
Again, we already do this in our example program.
We integrate 

.. math::
   \int d\phi e^{-S}
   :label:

The weight function of the Gibbs ensemble is exponentially peaked where S is
small. Further, :math:`S`
is an extensive quantity, meaning it grows with the volume.
The function gets more peaked as the volume increases.
If we use completely random points in the Monte Carlo
estimate, most of the values will be very small and only a small
fraction will actually contribute to the integral.

Instead we should choose the random points from a distribution and mimics
the function we are integrating. This leads to more measurements
actually contributing to the result and reduces the variance.

Say we are integrating 

.. math::
   I = \int dV f(x)
   :label:

We choose to draw random numbers from the distribution :math:`p(x)`.
The integral is

.. math::
   I = \int dV p(x) \frac{f(X)}{p(x)} 
     = \lim_{N\to\infty} \frac 1N \sum_i \frac{f(x_i)}{p(x)}
   :label:

If :math:`p` is chosen well, the function :math:`p/f` will be flatter
than the original and the variance of the result will be smaller. The 
optimal choice would be to use :math:`p(x) = C |f(x)|`.

Of course if it was that straightforward, we would always just draw from
the f(x). The problem is that f(x) is probably not a distribution, and to turn
it into one, we would need to calculate

.. math::
   C = \int dV f(x),
   :label:
which is exactly the problem we are trying to solve.

In lattice simulations though, this is exactly what we do. 
We are actually calculating

.. math::
   \ev{O} = \frac {\int [d\phi] O(\phi) e^{-S(\phi)} }{\int [d\phi] e^{-S(\phi)}}
   :label:

If we normalize the two integrals in the same way, the normalization drops out.

Since the denominator does not depend on :math:`O`,
the best choice generally is

.. math::
   g(\phi) = C \int [d\phi] e^{-S(\phi)}
   :label:

with an unknown normalization coefficient :math:`C`.
We then find

.. math::
   \ev{O} = \frac 1N \sum_i O(\phi_i)
   :label:

Notice that since the normalization coefficient drops out in the
Monte Carlo, we cannot calculate the partition function
directly from the simulation.


Autocorrelation Time
----------------------

The autocorrelation time measures the number of update steps required
to reach an independent configuration.

Given a measurement :math:`X`, let's label the individual measurements
as :math:`X_i` with the configuration numbers :math:`i=1,2,...,N`.
The autocorrelation function for measurable :math:`X` is the
correlation between measurements separated by time :math:`t`
normalized by the overall variation:

.. math::
   C(t) = \frac{\frac{1}{N-t} \sum_{i=1}^{N-t} X_i X_{i+t} - \ev{X}^2 }{\ev{X^2} - \ev{X}^2}
   :label:

This is normalized to :math:`C(0)=1`.

When the number of measurements is large, :math:`N\to \infty`, and
the time :math:`t` is not small (:math:`t\to\infty`, :math:`t\ll N`),
the autocorrelation function decays exponentially

.. math::
   C(t) \sim e^{-t/\tau_{exp}}
   :label:

The exponential autocorrelation time :math:`\tau_{exp}` is generally 
roughly the same for all measurements and really measures the time
it takes to decorrelate the configurations. For error analysis the relevant quantity is the integrated autocorrelation time

.. math::
   \tau_{int} = \frac 12 + \sum_{t=1}^\infty C(t)
   :label:

Note that if the autocorrelation function is purely exponential,
:math:`C(t) = e^{-t/\tau_{exp}}`, the two autocorrelation times are
the same, :math:`\tau_{int}\sim\tau_{exp}`. Generally 
:math:`\tau_{int} < \tau_{exp}`.

In a Monte Carlo simulation, the corrected error estimate is

.. math::
   \sigma_X = \sqrt{2\tau_{int}} \sigma_{X, naive}
   = \sqrt{2\tau_{int} \frac{ \sum_i \left(X_i - \ev{X}\right)^2 }{N(N-1)} }
   :label:

In practice :math:`N` is always finite. Taking this into account, the 
formula for the autocorrelation function is modified slightly:

.. math::
   &C(t) = \frac{\frac{1}{N-t} \sum_{i=1}^{N-t} X_i X_{i+t} - \ev{X}_1 \ev{X}_2 }{\ev{X^2} - \ev{X}^2},\\
   &\ev{X}_1 = \frac{1}{N-t} \sum_{i=1}^{N-t} X_i, \textrm{ and}\\
   &\ev{X}_2 = \frac{1}{N-t} \sum_{i=t}^{N} X_i
   :label:


It is important to keep track of autocorrelation times and to use them
in error estimates. Calculating observables too often is a waste of time,
so if the autocorrelation time is large, it might be worth adjusting the
number of updates between measurements.


.. hint::
   Check out jackknife blocking for a second possible error analysis method.
   It is useful when doing complicated calculations with expectation
   values and tracking errors becomes unreliable. It also works well
   with correlated data.


Update Sweeps
---------------

In our previous example we update the sites in random order. It is 
often better, in terms of the autocorrelation time, to update each
spin in order.

Here are three common choices for update ordering:

1. **Typewriter:** Just start site zero, update every site in a row
   in order and then continue to the next row. Simple to implement, but
   has a significant drawback: it breaks detailed balance in the Ising
   model, at least in 1D.

2. **Checkerboard:** When all interactions are nearest neighbor, it is
   possible to update all even sites (:math:`\sum_i x_i` is even) or all
   odd sites without problems with detailed balance. So for a full
   update run over even sites first and odd sites second.

3. **Random ordering:** Just pick sites completely at random. This 
   always respects detailed balance, but might leave a region unchanged
   for a long time.


Cluster and Worm Algorithm
------------

Local updates are not the only option. The most efficient method for
simulating the Ising model is a cluster algorithm. The Wolff cluster 
update [U. Wolff, PRL 62 (1989) 361] proceeds as follows:

1. Choose a random site i.
2. Check each neighboring site j.  If :math:`s_j==s_i`, join the site
   to the cluster with probability :math:`p=1-e^{-2\beta}`.
3. Repeat step 2 for each site :math:`j` that was joined to the cluster.
   Continue until there are no more sites to consider.
4. Flip all spins in the cluster.

Cluster updates perform an update on a large region at once. This way it can quickly 
go from a state where most spins are :math:`+1` to a state where most spins are
:math:`-1`. Doing this in multiple small steps is harder because each step can get
reversed.

Worm  updates are similar, but they update the configuration as they go.
They usually start by creating two mutually cancelling deformations 
(head and tail). The head moves around the lattice while updating the 
configuration. When the deformations meet, they cancel and the worm
update ends.
Worm algorithms can be efficient in a large :math:`\beta` expansion,
for example.

As a further generalization of the worm algorithm, it is possible to create
impossible configurations that, configuration with extra fields or zero weight,
as intermediate steps in an update. As long as the final result is not impossible,
the update works.




Gauge fields
=============

Gauge field theories are fundamental to the Standard Model of particle physics.
They describe the three fundamental interactions, strong and weak nuclear forces
and electromagnetism. In this section we will see how they can be implemented on
the lattice and study some of their properties.

Ising Gauge Model
-------------------

The defining feature of a gauge field is the gauge symmetry, a local symmetry of
the action. Let's look at the following action for spins :math:`s = \pm 1`

.. math::
   S = -\beta \sum_x \sum_{\mu>\nu} s_{x,\mu} s_{x+\mu,\nu} s_{x+\nu, \mu} s_{x,\nu}.
   :label:

The kinetic term consists of a product of spins around a single lattice square,
also known as a plaquette.

The action is symmetric with respect to the transformation

.. math::
   s_{x,\mu} \to -s_{x,\mu} \textrm{ for all } \mu.
   :label:

Now let's look at a scalar field

.. math::
   S = \sum_x \sum_\mu \frac {1}{a^2} \phi_x \left ( 2\phi_x - \phi_{x-\hat\mu} -\phi_{x+\hat\mu} \right ) .
   :label:

This model has a global symmetry, 

.. math::
   \phi_x \to -\phi_x \textrm{ for all } x.
   :label:

We turn this into a local symmetry by adding the Ising Gauge field and
multiplying with it in the derivative term.

.. math::
   S &= -\beta \sum_x \sum_{\mu>\nu} s_{x,\mu} s_{x+\mu,\nu} s_{x+\nu, \mu} s_{x,\nu}\\
     &+ \sum_x \sum_\mu \frac {1}{a^2} \phi_x \left ( 2\phi_x - s_{x-\hat\mu,\mu}\phi_{x-\mu} - s_{x,\mu}\phi_{x+\mu} \right ).
   :label:

The local symmetry is 

.. math::
   &\phi_x \to -\phi_x \\
   &s_{x,\mu} \to -s_{x,\mu} \textrm{ for all } \mu 
   :label:

We also need to check the measure. Here :math:`ds \to ds |ds'/ds| = ds |-1| = ds` and similarly 
:math:`d\phi \to d\phi`.


Quantum Electrodynamics 
--------------------------

Consider a complex scalar field,

.. math::
   S = \sum_x \left [ \sum_\mu \frac {1}{a^2} \phi_x^* \left ( 2\phi_x - \phi_{x-\hat\mu} -\phi_{x+\hat\mu} \right ) + m^2\phi_x^*\phi_x \right ].
   :label:

This model has a :math:`U(1)` global symmetry, 

.. math::
   \phi_x \to e^{i\alpha} \phi_x \textrm{ for all } x.
   :label:

As above, this can be turned into a local symmetry by adding a gauge field

.. math::
   U_{x,\mu} = e^{i aA_{x,\mu}}.
Note the factor :math:`a` in the exponential. The field :math:`A` can be
identified with the vector potential and should have the units of energy.

We add a plaquette term to the Lagrangian and
add the field :math:`U_\mu` to the derivative term:

.. math::
   S &= -\beta \sum_x \sum_{\mu>\nu} Re U_{x,\mu} U_{x+\mu,\nu} U^*_{x+\nu, \mu} U^*_{x,\nu}\\
     &+ \sum_x \sum_\mu \frac {1}{a^2} \phi^*_x \left ( 2\phi_x - U^*_{x-\hat\mu,\mu}\phi_{x-\mu} - U_{x,\mu}\phi_{x+\mu} \right ).
   :label:

The local symmetry is

.. math::
   &\phi_x \to e^{i\alpha_x} \phi_x \\
   &U_{x,\mu} \to e^{i\alpha_x} U_{x,\mu} e^{-i\alpha_{x+\hat\mu}} \textrm{ for all } \mu 
   :label:



**The plaquette action**

This model has the correct symmetries for a scalar field interacting with the
electromagnetic field and it was constructed with the same principle.
Still, we should check that the plaquette actions reproduces the familiar
continuum gauge action.

First, what is the field :math:`U_{x,\mu}`?
The product :math:`\phi^*_xU_{x,\mu}\phi_{x+\mu}` is invariant, it does not
change with the local transformation. Therefore we can identify it with the
parallel transport. This also works for longer chains, for example,

.. math::
   \phi^*_{(0,0,0,0)}U_{(0,0,0,0),0}U_{(1,0,0,0),1}U_{(1,1,0,0),0}\phi_{(2,1,0,0)}
   :label:
is invariant.

In continuum the parallel transport is the line integral

.. math::
   e^{i \int_x^y A_{\mu}(x) \cdot dr},
   :label:
where :math:`A_\mu` is the vector potential, the integral runs along a contour
from :math:`x` to :math:`y` and :math:`dr` is the line element along this contour.

In our discrete setup, the gauge field is constant on the links between sites, and
the parallel transport between :math:`x` and :math`x+\hat\mu` is

.. math::
   U_{x,\mu} = e^{i \int_x^{x+\hat\mu} A_{\mu}(x) \cdot dx} = e^{iaA_{x,\mu}} 
   :label:


Closed loops of the gauge field are also invariant. In fact, these can
be seen as the parallel transport of a massless fermion,

.. math::
   \int d\phi_x \phi^*_{x}U_{x,\mu} U_{x+\mu,\nu} U^*_{x+\nu, \mu} U^*_{x,\nu}\phi_{x}
    = Tr \left [ U_{x,\mu} U_{x+\mu,\nu} U^*_{x+\nu, \mu} U^*_{x,\nu} \right ].
   :label:

One way of deriving the field strength tensor is to do a parallel transport over a
closed loop and take the limit where the loop contracts to a point. This is 
what we will do here. The plaquette is the smallest closed loop possible on the
lattice and we will in the end take the limit :math:`a\to0`, where the loop
contracts to zero.

More formally, we can write

.. math::
   L_x &= -\beta Re \sum_{\mu>\nu} U_{x,\mu} U_{x+\mu,\nu} U^*_{x+\nu, \mu} U^*_{x,\nu}\\
   &= -\beta Re \sum_{\mu>\nu} e^{iaA_{x,\mu} + iaA_{x+\mu,\nu}-iaA_{x+\nu, \mu}-iaA_{x,\nu}}\\
   :label:

Now expanding

.. math::
   A_{x+\hat\mu,\nu} = A_{x,\mu} + a\partial_\mu A_{x,\nu} 
   + \frac 12 a^2 \partial^2_\mu A_{x,\nu} + \dots
   :label:

we find

.. math::
   L_x &= -\beta Re \sum_{\mu>\nu} e^{ia^2\partial_\nu A_{x,\mu} - ia\partial_\mu A_{x,\nu} + O(a^3)}\\
   &=-\beta Re\left( 1 + ia^2F_{x,\mu,\nu} - a^4F_{x,\mu,\nu} F_{x,\mu,\nu} \right )\\
   &=\beta \left ( -1 + a^4F_{x,\mu,\nu} F_{x,\mu,\nu} \right )
   :label: u1plaquetteaction

While constant shifts in the action do not actually make a difference, it is common
to use the action

.. math::
   L_x = \beta\left [ 1 - Re \sum_{\mu>\nu} U_{x,\mu} U_{x+\mu,\nu} U^*_{x+\nu, \mu} U^*_{x,\nu} \right ]
   :label:

This is the Wilson plaquette gauge action [K. Wilson, 1974] for a U(1) gauge theory,
such as electromagnetism.

The continuum action you may have seen does not have the factor :math:`\beta`.
This is where the coupling comes in: we would usually write the covariant derivative as

.. math::
   D_\mu = \partial_\mu +i gA_\mu
   :label:
This would lead to the parallel transport

.. math::
   U_\mu = e^{i agA_{x,\mu}}.
   :label:

So the vector potential is scaled by a factor of :math:`g`. On the lattice it is more
straightforward to not incorporate the coupling to the matrix, but if we did
the rescaling, :math:`A_\mu \to gA_\mu`, we would find

.. math::
   L_{x, gauge} \to \beta g^2 a^4 F_{x,\mu,\nu} F_{x,\mu,\nu}.
   :label: 

From here we can set :math:`\beta=\frac {1}{g^2}` to recover the continuum action.

So the action matches the continuum action for gauge fields. The propagators can
be derived similarly to the scalar fields. The kinetic terms are squared derivatives
and will produce the same dispersion relation.

In the Standard Model, the electromagnetic field interacts with fermion and the
Higgs field. We will get to fermions after we the gauge field section. The 
Higgs field also interacts with the weak gauge field and is not just a complex field.
Next we will look at :math:`SU(2)` interactions, such as the weak interaction,
and :math:`SU(3)` interactions, such as the strong interaction.
The results for :math:`SU(3)` generalize to any :math:`SU(N)`.

The electromagnetic interaction has a further complication not shared by the other
fundamental interactions. The electromagnetic couplings for each particle can be 
different. Modeling this on the lattice would require actually adding the coupling
to the parallel transport. For particle :math:`i`,

.. math::
   U_{\mu,i} = e^{i ag_i A_{x,\mu}} = U_\mu^{g_i}.
   :label:

In this case it is actually simpler to use store the vector potential
:math:`A_{x_\mu}` instead of the parallel transport :math:`U_{x_\mu}`.
The parallel transport can be calculated relatively easily,

.. math::
   U_{\mu,i} = e^{i a g_i A_{x,\mu}} = \cos(a g_i A_{x,\mu}) + i\sin(a g_i A_{x,\mu}).
   :label:

The exponentiation is not as straightforward in :math:`SU(N)`.


Non-Abelian Gauge Fields 
--------------------------

The U(1) gauge field is Abelian, meaning that

.. math::
   A_{x,\mu} A_{y,\nu} = A_{y,\nu} A_{x,\mu}
   :label:

The weak and strong nuclear forces are governed by a non-Abelian symmetry.
Again, starting from a scalar matter field :math:`\phi_x`, but this is now a vector
with :math:`N` indexes,

.. math::
   S = \sum_x \sum_\mu \frac {1}{a^2} \phi^\dagger_x \left ( 2\phi_x - \phi_{x-\hat\mu} - \phi_{x+\mu} \right ).
   :label:

the global symmetry is

.. math::
   &\phi_x \to \Lambda \phi_x \textrm{ for all } x \textrm{, and therefore } \\ 
   &\phi_x^\dagger \to \phi_x^\dagger \Lambda^\dagger \textrm{ for all } x\\
   &\Lambda^\dagger \Lambda = 1
   :label:

This in principle incorporates the U(1) symmetry, which is still a global symmetry.
It can be separated and have a different coupling, so we consider it a separate interaction.
This is achieved by requiring

.. math::
   \det \left ( \Lambda \right ) = 1.
   :label:

Thus the matrix :math:`\Lambda` is a member of the :math:`SU(N)` symmetry group.

The symmetry is made local using a gauge matrix :math:`U_{x,\mu}` as above.
The gauge invariant action is

.. math::
   S &= -\beta \sum_x \sum_{\mu>\nu}
   \left( 1 - \frac {1}{N} Re Tr U_{x,\mu} U_{x+\mu,\nu} U^*_{x+\nu, \mu} U^*_{x,\nu} \right )\\
     &+ \sum_x \sum_\mu \frac {1}{a^2} \phi^\dagger_x \left ( 2\phi_x - U^\dagger_{x-\hat\mu,\mu}\phi_{x-\hat\mu} - U_{x,\mu}\phi_{x+\mu} \right ).
   :label:


The gauge matrix is a special unitary matrix,

.. math::
   &\det \left ( U_{x,\mu} \right ) = 1 \\
   &U_{x,\mu}^\dagger U_{x,\mu} = 1
   :label:

and it can be written as

.. math::
   U = e^{i aA_{x,\mu}}.
   :label:

The color vector potential :math:`A_{x,\mu}` is a traceless, hermitean matrix, and it is a member
of a Lie algebra. The continuum action has the same form as in the U(1) case
(equation :eq:`u1plaquetteaction`),

.. math::
   L_x = Tr F_{x, \mu\nu}F_{x, \mu\nu}
   :label:



.. container:: note
   
   **Exercise**
   
   1. Check that the plaquette action generates the correct continuum action. 
   Note that it this case :math:`\left[ A_{x,\mu}, A_{y,\nu} \right] \neq 0`
   and that the definition of :math:`F_{\mu,\nu}` is a bit different.


**Note on Gauge Fixing**

Unlike in perturbative calculations, there is no reason to fix the gauge during a lattice simulation.
In general this is something we do not want to do.
The gauge symmetry simply adds an irrelevant dimension to the integral and reduces to a constant factor
in the metric.



Updating Gauge Fields
-----------------------

The action of the gauge field is still local and in fact each of the directional fields
:math:`U_\mu`, there are only nearest neighbor interactions. The complication is that we
need to suggest a new matrix in the correct group. For U(1), there are two options.
Either draw the vector potential and exponentiate it, or draw either the real
or imaginary part and calculate the other one. It is somewhat easier to use first
option:

 1. Loop over direction :math:`\mu`.
 2. Choose a site :math:`x`.
 3. Do a metropolis step:
   1. Calculate the sum of "staples",

   .. math::
      S_{x,\mu} = \sum_{\nu\neq\mu} U_{x+\mu,\nu} U^*_{x+\nu, \mu} U^*_{x,\nu}
      :label:
   2. Calculate the initial action 

      .. math::
         S_1 = Re U_{x,\mu} S_{x,\mu}
   3. Suggest the change 
   
      .. math::
         A'_{x,\mu} &= \left . A_{x,\mu} + C x \right |_{mod(2\pi)},\\
         U_{x,\mu} &= e^{iA_{x,\mu}} = \cos(A_{x,\mu}) + i \sin(A_{x,\mu})
         :label:
      (Here :math:`a=1`)


If you are also simulating a scalar matter field, you need to derivative term in the action,
since it depends on the gauge fields. This can be done by adding
:math:`-\phi^*_x\phi_{x+\mu}-\phi^*_{x+\mu}\phi_x` to the staple.

For non-Abelian fields, the vector potential is a matrix and exponentiating it is a bit more
complicated. One general option is to generate a "small" SU(N) matrix and multiply with it,
since this will produce an SU(N) matrix:

.. math::
   \Lambda &= e^{iCx\lambda_i} = \cos(C\sqrt{\tau_N}x)
    + i \sin(C\sqrt{\tau_N}x)\frac{\lambda_i}{\sqrt{\tau_N}}\\
   U'_{x,\mu} &= \Lambda U_{x,\mu}
   :label:

Here :math:`\lambda_i` is the generator matrix and :math:`\tau_N = \lambda_i^2` is a constant
that depends on how the generators are defined. For SU(2) with Pauli matrices
:math:`\tau_N = 1` and for SU(3) with Gell-Mann :math:`\tau_N = \frac23`.

While there are more efficient algorithms for each SU(N) group, this is a general method and
sufficient for our purposes.

During a simulation floating point errors will cause the gauge field to drift outside the SU(N)
group. In a long run this needs to be corrected for. The usual method is to change one of the
columns and the rows to make sure that the new matrix is Hermitean and has determinant :math:`1`.


Observables
---------------

Any observables constructed out of gauge dependent objects should be gauge invariant.
The gauge symmetry forces the expectation value of any non-invariant objects to zero
(Elitzur’s theorem). For example, :math:`\ev{U_{x,\mu}} = 0`


**Wilson Loops**

Consider static (infinitely massive) charge and a similar anticharge separated by the distance
:math:`R` that exist for the time :math:`T` and annihilate. To describe this we would need to parallel
transport a matter field :math:`\phi` from a point :math:`x` to :math:`x+(T,0)`, then to :math:`x+(T,R)`,
down in time (as an antiparticle) to :math:`x+(0,R)` and finally back to :math:`x` 
to annihilate. This can be described by the operator

.. math::
   W_{RT} &= \int [d\phi_{(0,0)} d\phi_{(R,T)}] \phi^\dagger_{(0,0)} \prod_{x=(0,R)}^{(0,0)} U_{x,\hat x}^\dagger \prod_{x=(T,R)}^{(0,R)} U_{x,\hat t}^\dagger \phi_{(T,R)} \\
   &\times \phi^\dagger_{(T,R)} \phi_{(0,0)} \prod_{x=(T,0)}^{(T,R)} U_{x,\hat x} \prod_{x=(0,0)}^{(T,0)} U_{x,\hat t} \phi_{(0,0)}\\
   &= Tr \prod_{x=(0,R)}^{(0,0)} U_{x,\hat x}^\dagger \prod_{x=(T,R)}^{(0,R)} U_{x,\hat t}^\dagger \prod_{x=(T,0)}^{(T,R)} U_{x,\hat x} \prod_{x=(0,0)}^{(T,0)} U_{x,\hat t}.
   :label:

This is known as a Wilson loop. Since it is a closed loop, it is a gauge invariant observable.
It has several possible interpretations, but perhaps the most straightforward is as the
propagator of a state of two static charged particles.
At large :math:`T` this should behave exponentially,

.. math::
   W_{RT} \to C e^{-V(R) T} \textrm{ as } T \to infty
   :label:
and indeed it generally does.

The Wilson loops allows us to probe wether a gauge theory is confining.
In a confining model, the potential energy :math:`V(R)` grows to infinity with distance,
whereas without confinement it remains finite.

.. math::
   &V(R\to\infty) \to \infty : \textrm{confinement}\\
   &V(R\to\infty) \to finite : \textrm{no confinement}
   :label:


The Wilson loop follows the
 - *perimeter law*: Free charges.

   :math:`W_{RT} \sim exp(-m (2R+2T))`, where m is the mass of the charge due to the gauge field.
   At constant :math:`R` and large :math:`T`, :math:`W_{RT} \sim exp(-2m T)` and :math:`V(R) = m`
 - *area law*: Confined charges

   :math:`W_{RT} \sim exp(-\sigma RT)`, :math:`V(R) = \sigma R`. String tension :math:`\sigma` ties
   the charges together.


Note that because of the exponential decay of the Wilson loop, measuring it becomes difficult when
:math:`R` and :math:`T` are large. The statistical uncertainty remains roughly constant and exponentially
more data is required. 

In general, the Wilson loop follows the phenomenological form

.. math::
   V(r) = V_0 + \sigma_r - \frac er + f\left[ G_L({\bf r}) - \frac 1r \right]


**Polyakov Line**

Now let's add a single static test charge. It does not annihilate,but moves forward in time
around the lattice. Since the lattice is periodic, this forms a closed loop:

.. math::
   P = Tr \prod_{t=0}^{T} U_{t,\hat t}
   :label:

Similarly to the above, the this corresponds to a propagator and scales exponentially,

.. math::
   \ev{P} = e^{-T F_Q}.
   :label:

Here :math:`F_Q` is the free energy of a single charged quark.
In a confining model charged quarks cannot appear alone and :math:`F_Q\to\infty`.
So we should find :math:`\ev{P}\to 0`. 

Both :math:`SU(2)` and :math:`SU(3)` have a deconfining transition and are not 
confining at high temperatures. Above some critical temperature we find :math:`\ev{P} \neq 0`.
The same result can of course be found using Wilson loops, but the Polyakov line provides
a particularly straightforward method.

The transition is in fact related to spin models in an interesting way. SU(N) fields have
an coincidental center symmetry. Center elements are elements of a group that commute with 
other elements. For SU(2) these are :math:`1` and :math:`-1` and for SU(3) they are
:math:`1`, :math:`exp(i2\pi/3)` and :math:`exp(i4\pi/3)`. Since they commute with all other
elements, we can multiply all links at a given :math:`t` with a center element and not affect
the plaquette action:

.. math::
   U_{t,({\bf x},\tau)} \to zU_{t,({\bf x},\tau)}  \textrm{ for all } {\bf x} \textrm{ and given } \tau.
   :label:

Now

.. math::
   L_{x,t,\nu} &= \beta \left [ 1 - ReTr U_{({\bf x}, \tau), t} U_{({\bf x}, \tau+1),\nu} U^\dagger_{({\bf x}+\hat\nu, \tau), t} U^\dagger_{({\bf x}, \tau),\nu} \right ]\\
   &\to \beta \left [ 1 - ReTr z U_{({\bf x}, \tau), t} U_{({\bf x}, \tau+1),\nu} z^*U^\dagger_{({\bf x}+\hat\nu, \tau), t} U^\dagger_{({\bf x}, \tau),\nu} \right ]\\
   &\to \beta \left [ 1 - ReTr z z^* U_{({\bf x}, \tau), t} U_{({\bf x}, \tau+1),\nu} U^\dagger_{({\bf x}+\hat\nu, \tau), t} U^\dagger_{({\bf x}, \tau),\nu} \right ]\\
   &\to \beta \left [ 1 - ReTr U_{({\bf x}, \tau), t} U_{({\bf x}, \tau+1),\nu} U^\dagger_{({\bf x}+\hat\nu, \tau), t} U^\dagger_{({\bf x}, \tau),\nu} \right ].
   :label:

However the Polyakov loop does not remain invariant. In only contains one link for each :math:`t`-layer, so

.. math::
   P \to zP.
   :label:

Since all configurations that differ by a center trasformation have the same weight, 
the expectation value is

.. math::
   \ev P = \frac 1N \sum_{n=0}^{N-1} e^{in\pi/N} \ev P = 0.
   :label:

The center symmetry transformations are similar to the symmetries of an :math:`Z_N` spin model
in 3 dimensions. The :math:`Z_3` part of the interaction in nearest neighbor and symmetry transformation cycles through the possible spins.

The most notable similarity is that the symmetry is never broken at finite volume.
The deconfinement transition requires large enough volume to prevent tunneling
between the sectors.

Any matter field breaks the center symmetry and prevents this mechanism for 
confinement. This is because dynamical particles can break the string: 
a particle-antiparticle loops can pop out of the vacuum and screen the
charges. Confinement is still happening but the Wilson loop does not measure
the correct thing.



Fermions
========

Naive Fermions
------------------

As mentioned above, we run into problems when discretizing fermions.
First let's just see what happens when we try to do this the most
straightforward way. The action of a free fermion field is

.. math::
   S = \int dx \sum_\mu \chi^\dagger(x) \gamma_\mu \partial_\mu \chi(x) - m\chi^\dagger(x)\chi(x)
   :label:

The fields :math:`\chi` are Grassmann numbers, so
:math:`\chi(x_1)\chi(x_2) = -\chi(x_2)\chi(x_1)`. This is true for any two
Grassmann numbers and :math:`\chi(x_1)\chi(x_1) = 0`.

The matrices :math:`\gamma_\mu` follow the anticommutation relation

.. math::
   \{\gamma_\mu,\gamma_\nu\} = 2 g_{\mu,\nu} = 2 \mathcal I
   :label:
 In 4 dimensions they are :math:`4\times 4` matrices.
The fields :math:`chi` are actually vectors in the space of these matrix indexes.

The Dirac equation in a kind of a square root of the Schrödinger equation in the sense that

.. math::
   (\gamma_\mu \partial_\mu - m)(\gamma_\nu \partial_\nu + m) = (\partial_\mu \partial_\mu + m^2) 
   :label:
It applies to a particle transforming in the spin :math:`1/2` representation of the Lorentz group.

From the point of view of discretization, however, the significant property is that there is only one
derivative. This will cause us some significant trouble.




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

-   Higgs-Yukawa model, transition

-   String tension in SU(3)

-   3D Thirring model (2D?)
