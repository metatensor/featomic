.. _userdoc-how-to-clebsch-gordan-products:

Clebsch-Gordan Products
=======================

In previous examples, namely `computing SOAP
<https://metatensor.github.io/featomic/latest/how-to/computing-soap.html>`_, `computing
lambda-SOAP
<https://metatensor.github.io/featomic/latest/how-to/computing-lambda-soap.html>`_, and
`computing density correlations
<https://metatensor.github.io/featomic/latest/how-to/computing-density-correlations.html>`_, we
used higher level calculators from the ``clebsch-gordan`` module to compute invariant
and equivariant descriptors.

In this tutorial, we will demonstrate the functionality of the lower-level
``ClebschGordanProduct`` calculator for performing general Clebsch-Gordan tensor
products to construct more complex descriptors. These will be shown through a series of
examples.

.. include:: ../examples/clebsch-gordan-products.rst
    :start-after: start-body
    :end-before: end-body