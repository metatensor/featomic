"""
Computing lambda SOAP features
=======================

.. start-body
"""


# Intro: 
# - SOAP in previous tutorial
# - Equation for CG contraction that gives lambda-SOAP
# - 

# Basic usage

# Choosing keys based on symmetry of target property
# (see also atomistic cookbook example on polariz)

# Show usage with two SphericalExpansions, comment 
# on using one SphEx and one LodeSphEx



# %%
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
# We can now define hyper parameters for the calculation

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
        "max_angular": 4,
        "radial": {"type": "Gto", "max_radial": 6},
    },
}

calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)