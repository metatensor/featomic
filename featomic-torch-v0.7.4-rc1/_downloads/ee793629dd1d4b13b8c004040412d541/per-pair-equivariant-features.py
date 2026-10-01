"""
Computing per-pair equivariant features
=======================================

.. start-body
"""

import chemfiles
from metatensor import Labels

from featomic import SphericalExpansion, SphericalExpansionByPair
from featomic.clebsch_gordan import EquivariantPowerSpectrumByPair


# %%
#
# Let's see how to compute the per-pair :math:`\lambda`-SOAP descriptor using Featomic.
#
# Read systems using Chemfiles. You can download the dataset for this example from our
# :download:`website <../../static/dataset.xyz>`.

with chemfiles.Trajectory("dataset.xyz") as trajectory:
    systems = [s for s in trajectory]

# %%
#
# Featomic also handles systems read by `ASE <https://wiki.fysik.dtu.dk/ase/>`_:
#
# ``systems = ase.io.read("dataset.xyz", ":")``.
#
# Next, define the hyperparameters for the spherical expansion:

HYPERPARAMETERS = {
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
# Create a spherical expansion and a spherical expansion by pair calculator.
# The :class:`~featomic.SphericalExpansion` and
# :class:`~featomic.SphericalExpansionByPair` classes use the hyperparameters above.
# Then, wrap them with :class:`~featomic.clebsch_gordan.EquivariantPowerSpectrumByPair`
# to compute the Clebsch-Gordan contraction for the per-pair :math:`\lambda`-SOAP.

spex_calculator = SphericalExpansion(**HYPERPARAMETERS)
per_pair_spex_calculator = SphericalExpansionByPair(**HYPERPARAMETERS)
calculator = EquivariantPowerSpectrumByPair(spex_calculator, per_pair_spex_calculator)

# %%
# Run the actual calculation

per_pair_power_spectrum = calculator.compute(systems, neighbors_to_properties=True)

# %%
# The result is a :class:`~metatensor.TensorMap` whose keys encode symmetry and the
# species of the atoms involved:

per_pair_power_spectrum.keys

# %%
# Often, you only need specific :math:`\lambda` values. For example, if the
# `target property is the Hamiltonian matrix on a minimal basis
# <https://tinyurl.com/ham-mat>`_, you can restrict the output to :math:`\lambda` values
# up to :math:`\lambda=2` using the ``selected_keys`` parameter:

per_pair_power_spectrum_minimal_basis = calculator.compute(
    systems,
    neighbors_to_properties=True,
    selected_keys=Labels.range("o3_lambda", 3),
)
per_pair_power_spectrum_minimal_basis.keys

# %%
# .. end-body
