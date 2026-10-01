.. _userdoc-how-to-computing-lambda-soap:

Computing :math:`\lambda`-SOAP features
=======================================

In a `previous example
<https://metatensor.github.io/featomic/latest/how-to/computing-soap.html>`_, we
computed the SOAP **scalar** descriptor. This tutorial focuses on the
equivariant generalization of SOAP to **tensorial** objects, commonly known as
:math:`\lambda`-SOAP. The key difference is that the SOAP scalar is the inner product
of two spherical expansions, whereas :math:`\lambda`-SOAP contracts two spherical
expansions using a Clebsch-Gordan coefficient:

.. math::

  \rho^{\otimes 2}_{z_1 z_2, n_1 n_2, l_1, l_2, \lambda \mu} (\{\mathbf{r}_i\}) =
  \sum_{m_1, m_2} C_{l_1 l_2, m_1 m_2}^{\lambda\mu}
  \rho_{z_1,n_1 l_1 m_1}(\{\mathbf{r}_i\}) \rho_{z_2,n_2 l_2 m_2}(\{\mathbf{r}_i\})

.. include:: ../examples/compute-lambda-soap.rst
    :start-after: start-body
    :end-before: end-body