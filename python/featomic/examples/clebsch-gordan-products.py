"""
Clebsch-Gordan Products
=======================

.. start-body
"""


# %%
#
# In order for operations to be gradient tracked, we will use a torch tensors as the
# arrays backend, and import required classes and utilities from ``featomic.torch`` and
# ``metatensor.torch``.

from typing import List

import chemfiles
import torch

import metatensor.torch as mts
from metatensor.torch.learn.nn import EquivariantLinear

from featomic.torch import SphericalExpansion, systems_to_torch
from featomic.torch.clebsch_gordan import ClebschGordanProduct, DensityCorrelations

# %%
# 
# Read systems using chemfiles. You can obtain the dataset used in this example from our
# :download:`website <../../static/dataset.xyz>`.

with chemfiles.Trajectory("dataset.xyz") as trajectory:
    systems = [s for s in trajectory]


# %%
#
# Example 1: computing a lambda-SOAP
# ==================================
#
# TODO!
#
# In a :ref:`previous tutorial <compute-lambda-soap.py>`, we saw how to use
# ``EquivariantPowerSpectrum`` to compute a lambda-SOAP descriptor. This class is a
# convenience wrapper around the more general ``ClebschGordanProduct``, and in this
# example we will demonstrate the core steps used to build a lambda-SOAP. 
# 
# This will be instructive for demonstrating the functionality needed to construct more
# complex descriptors.
#


# %%
#
# Example 2: contracted equivariants
# ==================================
#
# In a previous example on :ref:`computing density correlations
# <density-correlations.py>`, we computed arbitrary body-order equivariant descriptors
# using the ``DensityCorrelations`` class. As demonstrated, this results in the
# explosion of the feature dimension as higher body orders are reached.
#
# This example outlines how the size of the feature dimension can be controlled by
# combining feature contraction with the CG iteration steps. We will build a torch
# module that computes a learnable contracted bispectrum.
#
# First, define a function that computes the spherical expansion.

def compute_density(
    systems: List[chemfiles.Frame],
    hypers: dict,
    atom_types: List[int],
) -> mts.TensorMap:
    """
    Computes a spherical expansion (density) using featomic.

    ``neighbor_types`` is used to create a consistant feature size for the global atom
    types in the dataset.
    """
    spex_calc = SphericalExpansion(**hypers)
    density = spex_calc.compute(systems_to_torch(systems))
    density = density.keys_to_properties(
        keys_to_move=mts.Labels(
            ["neighbor_type"],
            torch.tensor(atom_types).reshape(-1, 1),
        )
    )

    return density

# %%
# 
# Use this function to compute the density, and inspect the output

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
atom_types = [1, 6, 7, 8]  # H, C, O, N
density = compute_density(systems, HYPER_PARAMETERS, atom_types)
print("density:", density)
print("density first block:", density[0])



# %%
#
# TODO!
#
# Build a torch module that DensityCorrelations

class ContractedEquivariantBispectrum(torch.nn.Module):

    def __init__(self, atom_types: List[int]) -> None:

        # TODO: initialize the calculators and linear layers
        return
    
    def forward(self, systems: List[chemfiles.Frame]) -> mts.TensorMap:
        """Computes bispectrum with learnable linear contractions."""

        # # compute density
        # density = compute_density(systems, self.atom_types)

        # # compute a lambda-SOAP with DensityCorrelations
        # lambda_soap = self.dens_corr_calc(density)

        # # contract the features
        # lambda_soap_contracted = self.linear_1(lambda_soap)

        # # compute the bispectrum
        # bispectrum = self.cg_product(
        #     lambda_soap_contracted, density
        # )

        # # contract once more
        # bispectrum_contracted = self.linear_2(bispectrum)

        return
 

# %%
# .. end-body