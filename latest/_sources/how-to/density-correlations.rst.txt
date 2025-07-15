.. _userdoc-how-to-density-correlations:

Computing density correlations
==============================

In a `previous example
<https://metatensor.github.io/featomic/latest/how-to/computing-lambda-soap.html>`_,
we computed the :math:`\lambda`-SOAP **equivariant** descriptor. This tutorial
focuses on the computation of arbitrary body-order descriptors by means of an
auto-correlations of spherical expansion (or 'density').

We will explore both :math:`\lambda`-SOAP --- a version of auto-correlations
with body-order 3 (taking into account one central atom and two neighbors) ---
an the bispectrum which is a representation with body-order 4.

.. include:: ../examples/density-correlations.rst
    :start-after: start-body
    :end-before: end-body
