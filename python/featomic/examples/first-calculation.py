"""
.. _userdoc-tutorials-get-started:

First descriptor computation
============================

This is an introduction to the featomic interface using a molecular crystals dataset
using the Python interface. If you are interested in another programming language we
recommend you first follow this tutorial and afterward take a look at the how-to guide
on :ref:`userdoc-how-to-computing-soap`.

The dataset
-----------

The atomic configurations used in our documentation are a small subset of the `ShiftML2
dataset <https://pubs.acs.org/doi/pdf/10.1021/acs.jpcc.2c03854>`_ containing molecular
crystals. There are four crystals - one with each of the elements ``[H, C]``, ``[H, C,
N, O]``, ``[H, C, N]``, and ``[H, C, O]``. For each crystal, we have 10 structures: the
first one is geometry-optimized. The following 9 contain atoms that are slightly
displaced from the geometry-optimized one. You can obtain the dataset from our
:download:`website <../../static/dataset.xyz>`.
"""

# %%
#
# We will start by importing all the required packages: the classic numpy;
# chemfiles to load data, and featomic to compute representations. Afterward
# we will load the dataset using chemfiles.

import chemfiles
import numpy as np

from featomic import SphericalExpansion


with chemfiles.Trajectory("dataset.xyz") as trajectory:
    structures = [s for s in trajectory]

print(f"The dataset contains {len(structures)} structures.")

# %%
#
# We will not explain here how to use chemfiles in detail, as we only use a few
# functions. Briefly, :class:`chemfiles.Trajectory` loads structure data in a format
# featomic can use. If you want to learn more about the possibilities take a look at the
# `chemfiles documentation <https://chemfiles.org>`_.
#
# Let us now take a look at the first structure of the dataset.

structure_0 = structures[0]

print(structure_0)

# %%
#
# With ``structure_0.atoms`` we get a list of the atoms that make up the first
# structure. The ``name`` attribute gives us the name of the specified atom.

elements, counts = np.unique(
    [atom.name for atom in structure_0.atoms], return_counts=True
)

print(
    f"The first structure contains "
    f"{counts[0]} {elements[0]}-atoms, "
    f"{counts[1]} {elements[1]}-atoms, "
    f"{counts[2]} {elements[2]}-atoms and "
    f"{counts[3]} {elements[3]}-atoms."
)

# %%
#
# Calculate a descriptor
# ----------------------
#
# We will now calculate an atomic descriptor for this structure using the SOAP spherical
# expansion as introduced by `Bartók, Kondor, and Csányi
# <http://dx.doi.org/10.1103/PhysRevB.87.184115>`_.
#
# To do so we define below a set of parameters telling featomic how the spherical
# expansion should be calculated. These parameters are also called hyper parameters
# since they are parameters of the representation, in opposition to parameters of
# machine learning models. Hyper parameters are a crucial part of calculating
# descriptors. Poorly selected hyper parameters will lead to a poor description of your
# dataset as discussed in the `literature <https://arxiv.org/abs/1502.02127>`_.

cutoff = {
    "radius": 4.5,
    "smoothing": {"type": "ShiftedCosine", "width": 0.5},
}

density = {
    "type": "Gaussian",
    "width": 0.3,
    "radial_scaling": {
        "type": "Willatt2018",
        "scale": 2.0,
        "rate": 1.0,
        "exponent": 4,
    },
}

basis = {
    "type": "TensorProduct",
    "max_angular": 5,
    "radial": {"type": "Gto", "max_radial": 8},
}

# %%
#
# After we set the hyper parameters we initialize a
# :class:`featomic.calculators.SphericalExpansion` object with hyper parameters defined
# above and run the :py:func:`featomic.calculators.CalculatorBase.compute()` method.

calculator = SphericalExpansion(cutoff=cutoff, density=density, basis=basis)
descriptor_0 = calculator.compute(structure_0)
print(type(descriptor_0))

# %%
#
# The descriptor format is a :class:`metatensor.TensorMap` object. Metatensor is like
# numpy for storing representations of atomistic ML data. Extensive details on the
# metatensor are covered in the `corresponding documentation
# <https://docs.metatensor.org>`_.
#
# We will now have a look at how the data is stored inside :class:`metatensor.TensorMap`
# objects.


print(descriptor_0)

# %%
#
# The :class:`metatensor.TensorMap` contains multiple :class:`metatensor.TensorBlock`.
# To distinguish them each block is associated with a unique key. For this example, we
# have one block for each angular channel labeled by ``o3_lambda`` (``o3_sigma`` is
# always equal to 1 since we only consider proper tensors, not pseudo-tensors), the
# central atom type ``center_type`` and neighbor atom type labeled by ``neighbor_type``.
# Different atomic types are represented using their atomic number, e.g. 1 for hydrogen,
# 6 for carbon, etc. To summarize, this descriptor contains 96 blocks covering all
# combinations of the angular channels, central atom type, and neighbor atom types in
# our dataset.
#
# Let us take a look at the second block (at index 1) in detail. This block contains the
# descriptor for the :math:`\lambda = 1` angular channel for hydrogen-hydrogen pairs.

block = descriptor_0.block(1)
print(descriptor_0.keys[1])


# %%
#
# Metadata about the descriptor
# -----------------------------
#
# The values of the representation are inside the blocks, in the ``block.values`` array.
# Each entry in this array is described by metadata carried by the block. For the
# spherical expansion calculator used in this tutorial the values have three dimensions
# which we can verify from the ``.shape`` attribute.


print(block.values.shape)

# %%
#
# The first dimension is described by the ``block.samples``, the intermediate dimension
# by ``block.components``, and the last dimension by ``block.properties``. The sample
# dimension has a length of eight because we have eight hydrogen atoms in the first
# structure. We can reveal more detailed metadata information about the sample dimension
# printing of the :py:attr:`metatensor.TensorBlock.samples` attribute of the block

print(block.samples)

# %%
#
# In these labels, the first column indicates the **structure**, which is 0 for all
# because we only computed the representation of a single structure. The second entry of
# each tuple refers to the index of the central **atom**.
#
# We can do a similar investigation for the second dimension of the value array,
# descrbied by :py:attr:`metatensor.TensorBlock.components`.

print(block.components)

# %%
#
# Here, the components are associated with the angular channels of the representation.
# The size of ``o3_mu`` is :math:`2l + 1`, where :math:`l` is the current ``o3_lambda``
# of the block. Here, its dimension is three because we are looking at the
# ``o3_lambda=1`` block. You may have noticed that the return value of the last call is
# a :class:`list` of :class:`metatensor.Labels` and not a single ``Labels`` instance.
# The reason is that a block can have several component dimensions as we will see below
# for the gradients.
#
# The last dimension represents the radial channels of spherical expansion. These are
# described in the :py:attr:`metatensor.TensorBlock.properties`:

print(block.properties)

# There is only one column here, named **n**, and indicating which of the radial channel
# we are looking at, from 0 to ``max_radial``.

# %%
#
# The descriptor values
# ---------------------
#
# After looking at the metadata we can investigate the actual data of the
# representation in more details

print(block.values[0, 0, :])

# %%
#
# By using ``[0, 0, :]`` we selected the first hydrogen and the first ``m`` channel. As
# you the output shows the values are floating point numbers, representing the
# coefficients of the spherical expansion.
#
# Featomic is also able to process more than one structure within one function call. You
# can process a whole dataset with

descriptor_full = calculator.compute(structures)

block_full = descriptor_full.block(0)
print(block_full.values.shape)

# %%
#
# Now, the 0th block of the :class:`metatensor.TensorMap` contains not eight but
# 420 entries in the first dimensions. This reflects the fact that in total we
# have 420 hydrogen atoms in the whole dataset.
#
# If you want to use another calculator instead of
# :class:`featomic.calculators.SphericalExpansion` shown here check out the
# :ref:`userdoc-references` section.
#
# Computing gradients
# -------------------
#
# Additionally, featomic is also able to calculate gradients on top of the
# values. Gradients are useful for constructing an ML potential and running
# simulations. For example ``gradients`` of the representation with respect to
# atomic positions can be calculated by setting the ``gradients`` parameter of
# the :py:func:`featomic.calculators.CalculatorBase.compute()` method to
# ``["positions"]``.

descriptor = calculator.compute(structure_0, gradients=["positions"])

block = descriptor.block(o3_lambda=1, center_type=1, neighbor_type=1)
gradient = block.gradient("positions")

print(gradient.values.shape)

# %%
#
# The calculated descriptor contains the values and in each block the associated
# position gradients as another :class:`metatensor.TensorBlock`, containing both
# gradient data and associated metadata.
#
# Compared to the features where we found three dimensions, gradients have four. Again
# the first is called ``samples`` and the ``properties``. The dimensions between the
# sample and property dimensions are denoted by ``components``.
#
# Looking at the shape in more detail we find that we have 52 samples, which is much
# more compared to features where we only have eight samples. This arises from the fact
# that we compute positions gradient for each samples in the features **with respect to
# the position of other atoms**. Since we are looking at the block with ``neighbor_type
# = 1``, only hydrogen neighbors will contribute to these gradients.
#
# The samples shows this in detail:

print(gradient.samples.print(max_entries=10))

# %%
#
# Here, the **sample** column indicate which of the sample of the features we are taking
# the gradients of; and (**system**,  **atom**) indicate which atom's positions we are
# taking the gradient with respect to. For example, looking at the gradient sample at
# index ``3``:

print("taking the gradient of", block.samples[gradient.samples[3]["sample"]])
print(
    "with respect to the position of system =",
    gradient.samples[3]["system"],
    "and atom =",
    gradient.samples[3]["atom"],
)

# %%
#
# Now looking at the components:

print(gradient.components)

# %%
#
# we find two of them. Besides the ``o3_mu`` component that is also present in the
# features position gradients also have a component indicating the direction of the
# gradient vector.
#
# Finally, the properties are the same as the features

print(gradient.properties)

# %%
#
# Featomic can also calculate gradients with respect to the strain (i.e. the virial).
# For this, you have to add ``"strain"`` to the list parsed to the ``gradients``
# parameter of the :py:func:`featomic.calculators.CalculatorBase.compute()` method.
# Strain gradients/virial are useful when computing the stress and the pressure.
#
# If you want to know about the effect of changing hypers take a look at the next
# tutorial. If you want to solve an explicit problem our :ref:`userdoc-how-to` might
# help you.
