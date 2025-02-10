"""
This module provides a convenience calculator for computing a two-center equivariant
power spectrum, or equivariant power spectrum by pair
"""

import json
from typing import List, Optional, Union

from . import _dispatch
from ._backend import (
    CalculatorBase,
    Device,
    DType,
    IntoSystem,
    Labels,
    TensorMap,
    TorchModule,
    operations,
)
from ._cg_product import ClebschGordanProduct
from ._density_correlations import _filter_redundant_keys


class EquivariantPowerSpectrumByPair(TorchModule):
    r"""
    Computes a general equivariant power spectrum by pair descriptor of two calculators,
    the second being a calculator by pair.

    Example
    -------

    As an example we calculate the equivariant power spectrum by pair for a spherical
    expansion and a spherical expansion by pair for a NaCl crystal.

    >>> import featomic
    >>> import ase

    Construct the NaCl crystal

    >>> atoms = ase.Atoms(
    ...     symbols="NaCl",
    ...     positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
    ...     pbc=True,
    ...     cell=[1, 1, 1],
    ... )

    Define the hyper parameters for the short-range spherical expansion

    >>> spex_hypers = {
    ...     "cutoff": {
    ...         "radius": 3.0,
    ...         "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    ...     },
    ...     "density": {
    ...         "type": "Gaussian",
    ...         "width": 0.3,
    ...     },
    ...     "basis": {
    ...         "type": "TensorProduct",
    ...         "max_angular": 2,
    ...         "radial": {"type": "Gto", "max_radial": 5},
    ...     },
    ... }

    Define the hyper parameters for the spherical expansion by pair

    >>> spex_by_pair_hypers = {
    ...     "cutoff": {
    ...         "radius": 5.0,
    ...         "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    ...     },
    ...     "density": {
    ...         "type": "Gaussian",
    ...         "width": 0.3,
    ...     },
    ...     "basis": {
    ...         "type": "TensorProduct",
    ...         "max_angular": 1,
    ...         "radial": {"type": "Gto", "max_radial": 5},
    ...     },
    ... }

    Construct the calculators

    >>> spex_calculator = featomic.SphericalExpansion(**spex_hypers)
    >>> spex_by_pair_calculator = featomic.SphericalExpansionByPair(
    ...     **spex_by_pair_hypers
    ... )

    Construct the power spectrum by pair calculators and compute the spherical expansion

    >>> calculator = featomic.clebsch_gordan.EquivariantPowerSpectrumByPair(
    ...     spex_calculator, spex_by_pair_calculator
    ... )
    >>> power_spectrum_by_pair = calculator.compute(atoms, neighbors_to_properties=True)

    The resulting equivariants are stored as :py:class:`metatensor.TensorMap` as for any
    other calculator. The keys contain the symmetry information:

    >>> power_spectrum_by_pair.keys
    Labels(
        o3_lambda  o3_sigma  first_atom_type  second_atom_type
            0         1            11                11
            1         1            11                11
            0         1            11                17
            1         1            11                17
            1         -1           11                11
            2         1            11                11
            1         -1           11                17
            2         1            11                17
            0         1            17                11
            1         1            17                11
            0         1            17                17
            1         1            17                17
            1         -1           17                11
            2         1            17                11
            1         -1           17                17
            2         1            17                17
    )

    The block properties contain the angular order of the combined blocks ("l_1",
    "l_2"), along with the neighbor type of the full spherical expansion and the radial
    channel indices.

    >>> power_spectrum_by_pair[0].properties.names
    ['l_1', 'l_2', 'neighbor_1_type', 'n_1', 'n_2']

    .. seealso::
        An equivariant power spectrum calculator for single-center descriptors can be
        found at :py:class:`featomic.clebsch_gordan.EquivariantPowerSpectrum`.
    """

    def __init__(
        self,
        calculator_1: CalculatorBase,
        calculator_2: CalculatorBase,
        neighbor_types: Optional[List[int]] = None,
        *,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
    ):
        """

        Constructs the equivariant power spectrum by pair calculator.

        :param calculator_1: first calculator that computes a density descriptor, either
            a :py:class:`featomic.SphericalExpansion` or
            :py:class:`featomic.LodeSphericalExpansion`.
        :param calculator_2: second calculator that computes a density by pair
            descriptor, it must be :py:class:`featomic.SphericalExpansionByPair` for
            now.
        :param neighbor_types: List of ``"neighbor_type"`` to use in the properties of
            the output. This option might be useful when running the calculation on
            subset of a whole dataset and trying to join along the ``sample`` dimension
            after the calculation. If ``None``, blocks are filled with
            ``"neighbor_type"`` found in the systems. This parameter is only used if
            ``neighbors_to_properties=True`` is passed to the :py:meth:`compute` method.
        :param dtype: the scalar type to use to store coefficients
        :param device: the computational device to use for calculations.
        """

        super().__init__()
        self.calculator_1 = calculator_1
        self.calculator_2 = calculator_2
        self.neighbor_types = neighbor_types
        self.dtype = dtype
        self.device = device

        supported_calculators_1 = ["lode_spherical_expansion", "spherical_expansion"]
        supported_calculators_2 = ["spherical_expansion_by_pair"]

        if self.calculator_1.c_name not in supported_calculators_1:
            raise ValueError(
                f"Only [{', '.join(supported_calculators_1)}] are supported for "
                f"`calculator_1`, got '{self.calculator_1.c_name}'"
            )

        parameters_1 = json.loads(calculator_1.parameters)

        # For the moment, only spherical expansion by pair is supported for calculator_2
        if self.calculator_2.c_name not in supported_calculators_2:
            raise ValueError(
                f"Only [{', '.join(supported_calculators_2)}] are supported for "
                f"`calculator_2`, got '{self.calculator_2.c_name}'"
            )

        parameters_2 = json.loads(calculator_2.parameters)
        if parameters_1["basis"]["type"] != "TensorProduct":
            raise ValueError("only 'TensorProduct' basis is supported for calculator_1")

        if parameters_2["basis"]["type"] != "TensorProduct":
            raise ValueError("only 'TensorProduct' basis is supported for calculator_2")

        self._cg_product = ClebschGordanProduct(
            max_angular=parameters_1["basis"]["max_angular"]
            + parameters_2["basis"]["max_angular"],
            cg_backend=None,
            keys_filter=_filter_redundant_keys,
            arrays_backend=None,
            dtype=dtype,
            device=device,
        )

    @property
    def name(self):
        """Name of this calculator."""
        return "EquivariantPowerSpectrumByPair"

    def compute(
        self,
        systems: Union[IntoSystem, List[IntoSystem]],
        selected_keys: Optional[Labels] = None,
        selected_samples: Optional[Labels] = None,
        neighbors_to_properties: bool = False,
    ) -> TensorMap:
        """
        Computes an equivariant power spectrum by pair, that can be thought of as a
        "Lambda-SOAP by pair" when doing a correlation of the SOAP density with a
        spherical expansion by pair.

        First computes a :py:class:`SphericalExpansion` density descriptor of body order
        2. Then a :py:class:`SphericalExpansionByPair` two-center density descriptor of
        body order 2 is computed.

        Before performing the Clebsch-Gordan tensor product, the spherical expansion
        density can be densified by moving the key dimension "neighbor_type" to the
        block properties. This is controlled by the ``neighbors_to_properties``
        parameter. Depending on the specific systems descriptors are being computed for,
        the sparsity of the spherical expansion can affect the computational cost of the
        Clebsch-Gordan tensor product.

        If ``neighbors_to_properties=True`` and ``neighbor_types`` have been passed to
        the constructor, property dimensions are created for all of these global atom
        types when moving the key dimension to properties. This ensures that the output
        properties dimension is of consistent size across all systems passed in
        ``systems``.

        Finally a single Clebsch-Gordan tensor product is taken to produce a body order
        3 equivariant power spectrum by pair.

        :param selected_keys: :py:class:`Labels`, the output keys to computed. If
            ``None``, all keys are computed. Subsets of key dimensions can be passed to
            compute output blocks that match in these dimensions.
        :param selected_samples: :py:class:`Labels`, Set of samples on which to run the
            calculation. Use ``None`` to run the calculation on all samples in
            the systems (this is the default). Gets passed to ``calculator_1``  and
            ``calculator_2``, therefore requiring that both calculators support sample
            selection.
        :param neighbors_to_properties: :py:class:`bool`, if true, densifies the
            spherical expansion by moving key dimension "neighbor_type" to properties
            prior to performing the Clebsch Gordan product step. Defaults to false.

        :return: :py:class:`TensorMap`, the output equivariant power spectrum by pair.
        """
        return self._equivariant_power_spectrum_by_pair(
            systems=systems,
            selected_keys=selected_keys,
            selected_samples=selected_samples,
            neighbors_to_properties=neighbors_to_properties,
            compute_metadata=False,
        )

    def forward(
        self,
        systems: Union[IntoSystem, List[IntoSystem]],
        selected_keys: Optional[Labels] = None,
        selected_samples: Optional[Labels] = None,
        neighbors_to_properties: bool = False,
    ) -> TensorMap:
        """
        Calls the :py:meth:`compute` method.

        This is intended for :py:class:`torch.nn.Module` compatibility, and should be
        ignored in pure Python mode.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self.compute(
            systems=systems,
            selected_keys=selected_keys,
            selected_samples=selected_samples,
            neighbors_to_properties=neighbors_to_properties,
        )

    def compute_metadata(
        self,
        systems: Union[IntoSystem, List[IntoSystem]],
        selected_keys: Optional[Labels] = None,
        selected_samples: Optional[Labels] = None,
        neighbors_to_properties: bool = False,
    ) -> TensorMap:
        """
        Returns the metadata-only :py:class:`TensorMap` that would be output by the
        function :py:meth:`compute` for the same calculator under the same settings,
        without performing the actual Clebsch-Gordan tensor products in the second step.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self._equivariant_power_spectrum_by_pair(
            systems=systems,
            selected_keys=selected_keys,
            selected_samples=selected_samples,
            neighbors_to_properties=neighbors_to_properties,
            compute_metadata=True,
        )

    def _equivariant_power_spectrum_by_pair(
        self,
        systems: Union[IntoSystem, List[IntoSystem]],
        selected_keys: Optional[Labels],
        selected_samples: Optional[Labels],
        neighbors_to_properties: bool,
        compute_metadata: bool,
    ) -> TensorMap:
        """
        Computes the equivariant power spectrum by pair, either fully or just metadata
        """
        # Compute density
        density_1 = self.calculator_1.compute(
            systems, selected_samples=selected_samples
        )
        # Rename "center_type" dimension to match "first_atom_type"
        density_1 = operations.rename_dimension(
            density_1, "keys", "center_type", "first_atom_type"
        )
        # Rename "neighbor_type" dimension so they are correlated
        density_1 = operations.rename_dimension(
            density_1, "keys", "neighbor_type", "neighbor_1_type"
        )
        # Rename samples to match the pair
        density_1 = operations.rename_dimension(
            density_1, "samples", "atom", "first_atom"
        )
        # Rename properties so they are correlated
        density_1 = operations.rename_dimension(density_1, "properties", "n", "n_1")

        # Compute pair density
        if selected_samples is not None:
            if "atom" in selected_samples.names:
                new_names = selected_samples.names
                new_names[selected_samples.names.index("atom")] = "first_atom"
                selected_samples = Labels(new_names, selected_samples.values)
        density_2 = self.calculator_2.compute(
            systems, selected_samples=selected_samples
        )

        # Rename properties so they are correlated
        density_2 = operations.rename_dimension(density_2, "properties", "n", "n_2")

        if neighbors_to_properties:
            if self.neighbor_types is None:  # just move neighbor type
                keys_to_move_1 = "neighbor_1_type"
            else:  # use the user-specified types
                values = _dispatch.list_to_array(
                    array=density_1.keys.values,
                    data=[[t] for t in self.neighbor_types],
                )
                keys_to_move_1 = Labels(names="neighbor_1_type", values=values)

            density_1 = density_1.keys_to_properties(keys_to_move_1)

        # Compute the power spectrum
        if compute_metadata:
            pow_spec = self._cg_product.compute_metadata(
                tensor_1=density_1,
                tensor_2=density_2,
                o3_lambda_1_new_name="l_1",
                o3_lambda_2_new_name="l_2",
                selected_keys=selected_keys,
            )
        else:
            pow_spec = self._cg_product.compute(
                tensor_1=density_1,
                tensor_2=density_2,
                o3_lambda_1_new_name="l_1",
                o3_lambda_2_new_name="l_2",
                selected_keys=selected_keys,
            )

        # Move the CG combination info keys to properties
        pow_spec = pow_spec.keys_to_properties(["l_1", "l_2"])

        return pow_spec
