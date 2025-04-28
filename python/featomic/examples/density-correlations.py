"""
Computing density correlations
==============================

.. start-body
"""

import chemfiles
import metatensor as mts
import numpy as np

from featomic import SphericalExpansion
from featomic.clebsch_gordan import DensityCorrelations, EquivariantPowerSpectrum


# %%
#
# Compute the spherical expansion
# -------------------------------
#
# Read systems using chemfiles. You can obtain the dataset used in this
# example from our :download:`website <../../static/dataset.xyz>`.

with chemfiles.Trajectory("dataset.xyz") as trajectory:
    systems = [s for s in trajectory]

# %%
#
# Featomic can also handles systems read by `ASE
# <https://wiki.fysik.dtu.dk/ase/>`_ using
#
# ``systems = ase.io.read("dataset.xyz", ":")``.
#
# We can now define the spherical expansion hyper parameters and compute the density.
# This will be used throughout the remainder of the tutorial.

HYPER_PARAMETERS = {
    "cutoff": {
        "radius": 5.0,
        "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.3,
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": 3,
        "radial": {"type": "Gto", "max_radial": 4},
    },
}

# Initialize the SphericalExpansion calculator and compute the density
spex_calc = SphericalExpansion(**HYPER_PARAMETERS)
density = spex_calc.compute(systems)

# %%
#
# Move to "neighbor_type" to properties, i.e. remove sparsity in this dimension.
#
# Note: this method should be called with care when computing on a subset of systems
# from a larger dataset. If the ``systems`` being computed contain a subset of the
# global atom types, an inconsistent feature dimension will be created. In this case,
# the argument "keys_to_move" should be specified as a ``Labels`` object with all global
# atom types.

density = density.keys_to_properties("neighbor_type")

# average number of features per block
print("total number of features:", np.sum([len(block.properties) for block in density]))

# %%
#
# Density correlations to build a lambda-SOAP
# -------------------------------------------
#
# Now we can use the DensityCorrelations calculator and specify that we want to take a
# single CG tensor product, i.e. ``n_correlations=1``.
#
# Initializing the calculator computes and stores the CG coefficients. As the density
# expansion is up to ``o3_lambda=3`` and we are doing a single contraction, we need CG
# coefficients computed up to ``o3_lambda=6`` in order to do a full contraction. Hence,
# we set ``max_angular=6``.

dens_corr_calc = DensityCorrelations(
    n_correlations=1,
    max_angular=6,
)

# %%
#
# This outputs an equivariant power spectrum descriptor of body-order 3, i.e.
# "lambda-SOAP" features.

lambda_soap = dens_corr_calc.compute(density)

# %%
#
# This is not quite equivalent to the result seen in the previous tutorial on
# :ref:`computing lambda-SOAP <compute-lambda-soap.py>`. The keys contain dimensions
# ``"l_1"`` and ``"l_2"`` which for a given block track the angular order of the blocks
# combined to create it.
#
# Keeping these dimensions in the keys is useful to allow for further CG products to be
# taken, building more complex descriptors. This is covered in a later tutorial on the
# :ref:`use of ClebschGordanProduct <clebsch-gordan-product>`.
#
# For now, we can move these key dimensions to properties. Inspect the metadata before
# and after moving these dimensions.

print("lambda-SOAP before keys_to_properties:", lambda_soap)
print("lambda-SOAP before keys_to_properties, first block:", lambda_soap[0])

lambda_soap = lambda_soap.keys_to_properties(["l_1", "l_2"])

print("lambda-SOAP after keys_to_properties:", lambda_soap)
print("lambda-SOAP after keys_to_properties, first block:", lambda_soap[0])

# %%
#
# This obtains a result that is equivalent to the lambda-SOAP seen in the previous
# tutorial. We can confirm this by computing an ``EquivariantPowerSpectrum`` and
# checking for consistency.

pow_spec_calc = EquivariantPowerSpectrum(spex_calc)
equi_pow_spec = pow_spec_calc.compute(systems, neighbors_to_properties=True)

assert mts.equal(lambda_soap, equi_pow_spec)

# %%
#
# Computing the bispectrum
# ========================
#
# Higher body order descriptors can be computed by increasing the ``n_correlations``
# parameter. The ``max_angular`` should also be increased to account for the increased
# combinations in angular momenta.
#
# With more iterations, the cost of the computation scales unfavourably. Let's use a
# density with small hyper parameters to demonstrate calculation of the bispectrum, a
# body-order 4 equivariant descriptor.

HYPER_PARAMETERS_SMALL = {
    "cutoff": {
        "radius": 5.0,
        "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.3,
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": 2,
        "radial": {"type": "Gto", "max_radial": 2},
    },
}

# %%
#
# Taking two CG combinations of a density expanded to ``o3_lambda=2`` requires CG
# coefficients computed up to ``max_angular=6``. This is given by ``(n_iterations + 1) *
# max_angular_density``.

# Initialize the SphericalExpansion calculator and compute the density
spex_calc = SphericalExpansion(**HYPER_PARAMETERS_SMALL)
density = spex_calc.compute(systems)
density = density.keys_to_properties("neighbor_type")

# Initialize DensityCorrelations calculator
dens_corr_calc = DensityCorrelations(
    n_correlations=2,
    max_angular=6,
)

# Compute the bispectrum
bispectrum = dens_corr_calc.compute(density)

# %%
#
# Inspect the features of the bispectrum. There are now ``"neighbor_x_type"`` and
# ``"n_x"`` (which track the radial channel indices) dimensions created by the product
# of feature spaces of the 3 density blocks combined to make each bispectrum block.
#
# For each block, its key contains dimensions tracking the angular order of blocks
# combined to create it, namely ``["l_1", "l_2", "k_2", "l_3"]``. The ``"l_"``
# dimensions track the angular order of the blocks from the original density, while
# ``"k_"`` dimensions track the angular order of intermediate blocks.
#
# Let's look at an example. Take the block indexed by key:
#
# ``LabelsEntry(o3_lambda=3, o3_sigma=1, center_type=7, l_1=1, l_2=2, k_2=1, l_3=2)``
#
# This was created in the following way.
#
# First, a block from ``density`` of angular order ``l_1=1`` was combined with a block
# of order ``l_2=2``. Angular momenta coupling rules state that non-zero combinations
# can only be created for output blocks with order ``| l_1 - l_2 | <= k_2 <= | l_1 + l_2
# |``, corresponding to ``[1, 2, 3]``. In this case, a block of order ``k_2=1`` was
# created.
#
# Next, this intermediate block of order ``k_2=1`` was then combined with a block from
# the original density of order ``l_3=2``. This can again create combinations ``[1, 2,
# 3]``, and in this case has been combined to create the output angular order
# ``o3_lambda=3``.

print("bispectrum:", bispectrum)
print("bispectrum first block:", bispectrum[0])

# %%
#
# As before, move these symmetry keys to properties and inspect the metadata and the
# mean size of the features.

bispectrum = bispectrum.keys_to_properties(["l_1", "l_2", "k_2", "l_3"])

print("bispectrum:", bispectrum)
print("bispectrum first block:", bispectrum[0])
print(
    "total number of features:", np.sum([len(block.properties) for block in bispectrum])
)

# %%
#
# Instead of computing the full CG product, a threshold can be defined to limit the
# maximum angular order of blocks computed at each step in the iterative CG coupling
# steps.
#
# This is controlled by the ``angular_cutoff`` parameter, and allows us to initialize
# the calculator with a lower ``max_angular``.
#
# Note that any truncation of the angular channels away from the maximal allowed by
# angular momenta coupling rules results in some loss of information.
#
# Let's truncate to an angular cutoff of 3:

angular_cutoff = 3
dens_corr_calc = DensityCorrelations(
    n_correlations=2,
    max_angular=angular_cutoff,
)
bispectrum_truncated = dens_corr_calc.compute(density, angular_cutoff=angular_cutoff)

# Move the "l_" and "k_" keys to properties
bispectrum_truncated = bispectrum_truncated.keys_to_properties(
    ["l_1", "l_2", "k_2", "l_3"]
)

print("truncated bispectrum:", bispectrum_truncated)
print("truncated bispectrum first block:", bispectrum_truncated[0])
print(
    "total number of features:",
    np.sum([len(block.properties) for block in bispectrum_truncated]),
)

# %%
#
# To compute a descriptor that matches the symmetry of a given target property, the
# ``selected_keys`` argument can be passed to the ``compute`` method. This was also seen
# in the previous tutorial on :ref:`computing lambda-SOAP <compute-lambda-soap.py>`.
#
# Following this example, to compute the truncated bispectrum for a polarizability
# tensor:
bispectrum_truncated_key_select = dens_corr_calc.compute(
    density,
    angular_cutoff=angular_cutoff,
    selected_keys=mts.Labels(
        ["o3_lambda", "o3_sigma"],
        np.array([[0, 1], [2, 1]]),
    ),
)

# Move the "l_" and "k_" keys to properties
bispectrum_truncated_key_select = bispectrum_truncated_key_select.keys_to_properties(
    ["l_1", "l_2", "k_2", "l_3"]
)

print("truncated bispectrum with selected keys:", bispectrum_truncated_key_select)
print(
    "total number of features:",
    np.sum([len(block.properties) for block in bispectrum_truncated_key_select]),
)


# %%
#
# Conclusion
# ==========
#
# ``DensityCorrelations`` can be used to build equivariants of arbitrary body order from
# a spherical expansion of decorated atomic point cloud data.
#
# A key limitation of this approach is an exploding feature size. To reduce the number
# of output blocks, the ``angular_cutoff`` parameter can be used. This can be controlled
# by introducing contractions after each CG combination in the iteration. This will be
# covered in the next tutorial, :ref:`Clebsch-Gordan products
# <clebsch-gordan-products.py>`.

# %%
# .. end-body
