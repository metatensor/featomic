.. _userdoc-how-to-density-correlations:

Computing density correlations
==============================

In a `previous example
<https://metatensor.github.io/featomic/latest/how-to/computing-lambda-soap.html>`_, we
computed the lambda-SOAP **equivariant** descriptor. This tutorial focuses on the
computation of arbitrary body-order descriptors by means of an auto-correlations of
spherical expansion (or 'density'). Lambda-SOAP is the body-order 3 version of this, but
we will also explore the bispectrum, and how to combine iterative Clebsch-Gordan tensor
products with feature space contractions.

.. include:: ../examples/density-correlations.rst
    :start-after: start-body
    :end-before: end-body