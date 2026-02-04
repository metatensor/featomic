"""
.. _compute-lambda-soap:

Computing  λ-SOAP features
==========================

.. start-body
"""

import chemfiles
import numpy as np
from metatensor import Labels

from featomic import LodeSphericalExpansion, SphericalExpansion
from featomic.clebsch_gordan import EquivariantPowerSpectrum


# %%
#
# Let's see how to compute the :math:`\lambda`-SOAP descriptor using featomic.
#
# First we can read the input systems using chemfiles. You can download the dataset for
# this example from our :download:`website <../../static/dataset.xyz>`.

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
# Create the spherical expansion calculator. The :class:`~featomic.SphericalExpansion`
# class uses the hyperparameters above. Then, wrap it with
# :class:`~featomic.clebsch_gordan.EquivariantPowerSpectrum` to compute the
# Clebsch-Gordan contraction for :math:`\lambda`-SOAP.

spex_calculator = SphericalExpansion(**HYPERPARAMETERS)
calculator = EquivariantPowerSpectrum(spex_calculator)

# %%
# Run the actual calculation

power_spectrum = calculator.compute(systems, neighbors_to_properties=True)

# %%
# The result is a :class:`~metatensor.TensorMap` whose keys encode symmetry:

power_spectrum.keys

# %%
# Often, you only need specific :math:`\lambda` values. For example, if the
# `target property is the polarizability tensor
# <https://atomistic-cookbook.org/examples/polarizability/polarizability.html>`_,
# (a rank-2 symmetric Cartesian tensor), you can restrict the output to
# :math:`\lambda=0` and :math:`\lambda=2` (with :math:`\sigma=1` to discard
# `pseudotensors <https://en.wikipedia.org/wiki/Pseudotensor>`_) using the
# ``selected_keys`` parameter:

power_spectrum_0_2 = calculator.compute(
    systems,
    neighbors_to_properties=True,
    selected_keys=Labels(["o3_lambda", "o3_sigma"], np.array([[0, 1], [2, 1]])),
)
power_spectrum_0_2.keys

# %%
# You can also compute a :math:`\lambda`-SOAP-like descriptor using two different
# expansions. For instance, combine a standarad spherical expansion with a long-range
# :class:`~featomic.LodeSphericalExpansion`:

LODE_HYPERPARAMETERS = {
    "density": {
        "type": "SmearedPowerLaw",
        "smearing": 0.3,
        "exponent": 1,
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": 2,
        "radial": {"type": "Gto", "max_radial": 3, "radius": 1.0},
    },
}
lode_calculator = LodeSphericalExpansion(**LODE_HYPERPARAMETERS)
calculator = EquivariantPowerSpectrum(spex_calculator, lode_calculator)
power_spectrum = calculator.compute(systems, neighbors_to_properties=True)
power_spectrum.keys

# %%
#
# .. end-body
