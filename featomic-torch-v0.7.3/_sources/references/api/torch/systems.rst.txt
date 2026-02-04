System
======

Instead of a custom ``System`` class, ``featomic-torch`` uses the class defined
by metatomic: :py:class:`metatomic.torch.System`. Featomic provides converters
from all the supported system providers (i.e. everything in
:py:class:`featomic.IntoSystem`) to the TorchScript compatible ``System``.

.. autofunction:: featomic.torch.systems_to_torch
