"""
This module contains utilities to convert cartesian TensorMap to spherical and
respectively.
"""

from typing import List, Optional

import numpy as np

from . import _coefficients, _dispatch
from ._backend import (
    Array,
    Labels,
    TensorBlock,
    TensorMap,
    TorchTensor,
    torch_jit_is_scripting,
)


def cartesian_to_spherical(
    tensor: TensorMap,
    components: List[str],
    keep_l_in_keys: Optional[bool] = None,
    remove_blocks_threshold: Optional[float] = 1e-9,
    cg_backend: Optional[str] = None,
    cg_coefficients: Optional[TensorMap] = None,
) -> TensorMap:
    """
    Transform a ``tensor`` of arbitrary rank from cartesian form to a spherical form.

    Starting from a tensor on a basis of product of cartesian coordinates, this function
    computes the same tensor using a basis of spherical harmonics ``Y^M_L``. For
    example, a rank 1 tensor with a single "xyz" component would be represented as a
    single L=1 spherical harmonic; while a rank 5 tensor using a product basis ``ℝ^3 ⊗
    ℝ^3 ⊗ ℝ^3 ⊗ ℝ^3 ⊗ ℝ^3`` would require multiple blocks up to L=5 spherical harmonics.

    A single :py:class:`TensorBlock` in the input might correspond to multiple
    :py:class:`TensorBlock` in the output. The output keys will contain all the
    dimensions of the input keys, plus ``o3_lambda`` (indicating the spherical harmonics
    degree) and ``o3_sigma`` (indicating that this block is a proper- or improper tensor
    with ``+1`` and ``-1`` respectively). If ``keep_l_in_keys`` is ``True`` or if the
    input tensor is a tensor of rank 3 or more, the keys will also contain multiple
    ``l_{i}`` and  ``k_{i}`` dimensions, which indicate which angular momenta have been
    coupled together in which order to get this block.

    ``components`` specifies which ones of the components of the input
    :py:class:`TensorMap` should be transformed from cartesian to spherical. All these
    components will be replaced in the output by a single ``o3_mu`` component,
    corresponding to the spherical harmonics ``M``.

    By default, symmetric tensors will only contain blocks corresponding to
    ``o3_sigma=1``. This is achieved by checking the norm of the blocks after the full
    calculation; and dropping any block with a norm below ``remove_blocks_epsilon``. To
    keep all blocks regardless of their norm, you can set
    ``remove_blocks_epsilon=None``.

    :param tensor: input tensor, using a cartesian product basis
    :param components: components of the input tensor to transform into spherical
        components
    :param keep_l_in_keys: should the output contains the values of angular momenta that
        were combined together? This defaults to ``False`` for rank 1 and 2 tensors,
        and ``True`` for all other tensors.

        Keys named ``l_{i}`` correspond to the input ``components``, with ``l_1`` being
        the last entry in ``components`` and ``l_N`` the first one. Keys named ``k_{i}``
        correspond to intermediary spherical components created during the calculation,
        i.e. a ``k_{i}`` used to be ``o3_lambda``.
    :param remove_blocks_threshold: Numerical tolerance to use when determining if a
        block's norm is zero or not. Blocks with zero norm will be excluded from the
        output. Set this to ``None`` to keep all blocks in the output.
    :param cg_backend: Backend to use for Clebsch-Gordan calculations. This can be
        ``"python-dense"`` or ``"python-sparse"`` for dense or sparse operations
        respectively. If ``None``, this is automatically determined.
    :param cg_coefficients: Cache containing Clebsch-Gordan coefficients. This is
        optional except when using this function from TorchScript. The coefficients
        should be computed with :py:func:`calculate_cg_coefficients`, using the same
        ``cg_backend`` as this function.

    :return: :py:class:`TensorMap` containing spherical components instead of cartesian
        components.
    """
    if len(tensor) == 0 or len(components) == 0:
        # nothing to do
        return tensor

    if not isinstance(components, list):
        raise TypeError(f"`components` should be a list, got {type(components)}")

    if keep_l_in_keys is None:
        if len(components) < 3:
            keep_l_in_keys = False
        else:
            keep_l_in_keys = True

    axes_to_convert: List[int] = []
    all_component_names = tensor.component_names
    for component in components:
        if component in all_component_names:
            idx = all_component_names.index(component)
            axes_to_convert.append(idx + 1)
        else:
            raise ValueError(f"'{component}' is not part of this tensor components")

    for key, block in tensor.items():
        for idx in axes_to_convert:
            values_list = _dispatch.to_int_list(block.components[idx - 1].values[:, 0])
            if values_list != [0, 1, 2]:
                name = block.components[idx - 1].names[0]
                raise ValueError(
                    f"component '{name}' in block for {key.print()} should have "
                    f"[0, 1, 2] as values, got {values_list} instead"
                )

    # we need components to be consecutive
    if list(range(axes_to_convert[0], axes_to_convert[-1] + 1)) != axes_to_convert:
        raise ValueError(
            f"this function only supports consecutive components, {components} are not"
        )

    key_names = tensor.keys.names
    if "o3_lambda" in key_names:
        raise ValueError(
            "this tensor already has an `o3_lambda` key, "
            "is it already in spherical form?"
        )

    for i in range(len(components)):
        if f"l_{i}" in key_names:
            raise ValueError(
                f"this tensor already has an `l_{i}` key, "
                "is it already in spherical form?"
            )

    tensor_rank = len(components)
    if tensor_rank > 2 and not keep_l_in_keys:
        raise ValueError(
            "`keep_l_in_keys` must be `True` for tensors of rank 3 and above"
        )

    if isinstance(tensor.block(0).values, TorchTensor):
        arrays_backend = "torch"
        values = tensor.block(0).values
        dtype = values.dtype
        device = values.device
    elif isinstance(tensor.block(0).values, np.ndarray):
        arrays_backend = "numpy"
        values = tensor.block(0).values
        dtype = values.dtype
        device = "cpu"
    else:
        raise TypeError(
            f"unknown array type in tensor ({type(tensor.block(0).values)}), "
            "only numpy and torch are supported"
        )

    if cg_backend is None:
        # TODO: benchmark & change the default?
        if arrays_backend == "torch":
            cg_backend = "python-dense"
        else:
            cg_backend = "python-sparse"

    new_component_names: List[str] = []
    for idx in range(len(tensor.component_names)):
        if idx + 1 in axes_to_convert:
            if len(axes_to_convert) == 1:
                new_component_names.append("o3_mu")
            else:
                new_component_names.append(f"__internal_to_convert_{idx}")
        else:
            new_component_names.append("")

    # Step 1: transform xyz dimensions to o3_lambda=1 dimensions
    # This is done with `roll`, since (y, z, x) is the same as m = (-1, 0, 1)
    new_blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        values = _dispatch.roll(
            block.values,
            shifts=[-1] * len(axes_to_convert),
            axis=axes_to_convert,
        )

        new_components: List[Labels] = []
        for idx, component in enumerate(block.components):
            if idx + 1 in axes_to_convert:
                new_components.append(
                    Labels(
                        names=[new_component_names[idx]],
                        values=_dispatch.int_array_like(
                            [[-1], [0], [1]], component.values
                        ),
                    )
                )
            else:
                new_components.append(component)

        new_blocks.append(
            TensorBlock(values, block.samples, new_components, block.properties)
        )

    if tensor_rank == 1:
        new_keys = tensor.keys
        # we are done, add o3_lambda/o3_sigma/l_1 to the keys & return
        if keep_l_in_keys:
            new_keys = new_keys.insert(
                0,
                "l_1",
                _dispatch.int_array_like([1] * len(new_keys), new_keys.values),
            )

        new_keys = new_keys.insert(
            0,
            "o3_sigma",
            _dispatch.int_array_like([1] * len(tensor.keys), tensor.keys.values),
        )

        new_keys = new_keys.insert(
            0,
            "o3_lambda",
            _dispatch.int_array_like([1] * len(tensor.keys), tensor.keys.values),
        )

        return TensorMap(new_keys, new_blocks)

    # Step 2: if there is more than one dimension, couple them with CG coefficients
    #
    # We start from an array of shape [..., 3, 3, 3, 3, 3, ...] with as many 3 as
    # `len(components)`. Then we iteratively combine the two rightmost components into
    # as many new lambda/mu entries as required, until there is only one component left.
    # Each step will create multiple blocks (corresponding to the different o3_lambda
    # created by combining two o3_lambda=1 terms), that might on their turn create more
    # blocks if more combinations are required.
    #
    # For example, with a rank 3 tensor we go through the following:
    #
    # - Step 1: [..., 3, 3, 3, ...] => [..., 3, 1, ...] (o3_lambda=0, o3_sigma=+1)
    #                               => [..., 3, 3, ...] (o3_lambda=1, o3_sigma=-1)
    #                               => [..., 3, 5, ...] (o3_lambda=2, o3_sigma=+1)
    #
    # - Step 2: [..., 3, 1, ...] => [..., 3, ...] (o3_lambda=1, o3_sigma=+1)
    #           [..., 3, 3, ...] => [..., 1, ...] (o3_lambda=0, o3_sigma=-1)
    #                            => [..., 3, ...] (o3_lambda=1, o3_sigma=+1)
    #                            => [..., 5, ...] (o3_lambda=2, o3_sigma=-1)
    #           [..., 3, 5, ...] => [..., 3, ...] (o3_lambda=1, o3_sigma=+1)
    #                            => [..., 5, ...] (o3_lambda=2, o3_sigma=-1)
    #                            => [..., 7, ...] (o3_lambda=3, o3_sigma=+1)

    # Use the rolled block values as the starting point for higher-rank tensors
    tensor = TensorMap(tensor.keys, new_blocks)

    if cg_coefficients is None:
        if torch_jit_is_scripting():
            raise ValueError(
                "in TorchScript mode, `cg_coefficients` must be pre-computed "
                "and given to this function explicitly"
            )
        else:
            cg_coefficients = _coefficients.calculate_cg_coefficients(
                lambda_max=len(axes_to_convert),
                cg_backend=cg_backend,
                arrays_backend=arrays_backend,
                dtype=dtype,
                device=device,
            )

    iteration_index = 0
    while len(axes_to_convert) > 1:
        tensor = _do_coupling(
            tensor=tensor,
            component_1=axes_to_convert[-2] - 1,
            component_2=axes_to_convert[-1] - 1,
            cg_coefficients=cg_coefficients,
            cg_backend=cg_backend,
            keep_l_in_keys=keep_l_in_keys,
            iteration_index=iteration_index,
        )

        axes_to_convert.pop()
        iteration_index += 1

    if remove_blocks_threshold is None:
        return tensor

    # Step 3: for symmetry reasons, some of the blocks will be zero everywhere (for
    # example o3_sigma=-1 blocks if the input tensor is fully symmetric). If the user
    # gave us a threshold, we remove all blocks with a norm below this threshold.
    new_keys_values: List[Array] = []
    new_blocks: List[TensorBlock] = []
    for key_idx, block in enumerate(tensor.blocks()):
        key = tensor.keys.entry(key_idx)
        values = block.values.reshape(-1, 1)
        norm = values.T @ values
        if norm > remove_blocks_threshold:
            new_keys_values.append(key.values.reshape(1, -1))
            new_blocks.append(
                TensorBlock(
                    values=block.values,
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
            )

    return TensorMap(
        Labels(tensor.keys.names, _dispatch.concatenate(new_keys_values, axis=0)),
        new_blocks,
    )


def spherical_to_cartesian(
    tensor: TensorMap,
    new_component_names: Optional[List[str]] = None,
    cg_backend: Optional[str] = None,
    cg_coefficients: Optional[TensorMap] = None,
) -> TensorMap:
    """
    Transform a ``tensor`` from spherical form back to cartesian form.

    This is the exact inverse of :py:func:`cartesian_to_spherical`. Starting from a
    tensor expressed on a basis of spherical harmonics ``Y^M_L`` (as produced by
    :py:func:`cartesian_to_spherical` with ``keep_l_in_keys=True`` and
    ``remove_blocks_threshold=None``), this function reconstructs the corresponding
    cartesian product-basis tensor.

    The input keys must contain ``o3_lambda`` and ``o3_sigma``, and the single ``o3_mu``
    component will be replaced by a number of cartesian components (according to the
    rank of the tensor, each with values ``[0, 1, 2]``). For tensors of rank 3 and above
    the keys must also contain the ``l_{i}`` and ``k_{i}`` dimensions describing the
    coupling history (these are added automatically by
    :py:func:`cartesian_to_spherical`). For rank 1 and 2 tensors, the missing ``l_{i}``
    dimensions are re-created internally and assumed to be 1.

    :param tensor: input tensor, using a spherical harmonics basis
    :param new_component_names: names to give to the reconstructed cartesian components.
        There should be as many of them as is the rank of the tensor. If ``None``, the
        components are named ``xyz_1``, ``xyz_2``, ...
    :param cg_backend: Backend to use for Clebsch-Gordan calculations. This can be
        ``"python-dense"`` or ``"python-sparse"``. If ``None``, this is automatically
        determined.
    :param cg_coefficients: Cache containing Clebsch-Gordan coefficients. This is
        optional except when using this function from TorchScript. The coefficients
        should be computed with :py:func:`calculate_cg_coefficients`, using the same
        ``cg_backend`` as this function.

    :return: :py:class:`TensorMap` containing cartesian components instead of spherical
        components.
    """
    if len(tensor) == 0:
        # nothing to do
        return tensor

    if torch_jit_is_scripting():
        use_torch = True
    else:
        if isinstance(tensor.block(0).values, TorchTensor):
            use_torch = True
        elif isinstance(tensor.block(0).values, np.ndarray):
            use_torch = False
        else:
            raise TypeError(
                f"unknown array type in tensor ({type(tensor.block(0).values)}), "
                "only numpy and torch are supported"
            )

    if cg_backend is None:
        # TODO: benchmark & change the default?
        if use_torch:
            cg_backend = "python-dense"
        else:
            cg_backend = "python-sparse"

    key_names = tensor.keys.names
    if "o3_lambda" not in key_names or "o3_sigma" not in key_names:
        raise ValueError(
            "this tensor is missing one of `o3_lambda` or `o3_sigma` in the keys, "
            "is it already in cartesian form?"
        )

    if "o3_mu" not in tensor.component_names:
        raise ValueError(
            "this tensor does not have an `o3_mu` component, "
            "is it already in cartesian form?"
        )

    new_keys = tensor.keys
    array_of_ones = _dispatch.int_array_like([1] * len(new_keys), new_keys.values)

    # add back l_1, l_2 keys if they are missing from a rank 1 or 2 tensor
    lambda_min = int(_dispatch.min(tensor.keys.column("o3_lambda")))
    lambda_max = int(_dispatch.max(tensor.keys.column("o3_lambda")))
    if lambda_min < 0:
        raise ValueError("expected all o3_lambda to be >= 0 in the keys")
    elif lambda_max < 1:
        raise ValueError("expected o3_lambda to got at least to 1 in the keys")
    elif lambda_max == 1:
        if "l_1" not in key_names:
            new_keys = new_keys.insert(0, "l_1", array_of_ones)
    elif lambda_max == 2:
        if "l_1" not in key_names:
            new_keys = new_keys.insert(0, "l_1", array_of_ones)
        if "l_2" not in key_names:
            new_keys = new_keys.insert(0, "l_2", array_of_ones)

    # check the l_x and k_x keys
    l_keys = []
    for n in range(1, lambda_max + 1):
        if f"l_{n}" not in key_names:
            raise ValueError(f"expected a l_{n} dimension in the keys")
        one = _dispatch.int_array_like([1], new_keys.values)
        column = _dispatch.unique(new_keys.column(f"l_{n}"))
        if not _dispatch.all(column == one):
            raise ValueError(f"expected only 1 as value for the l_{n} dimension")

        if n > 2:
            if f"k_{n - 2}" not in key_names:
                raise ValueError(f"expected a k_{n - 2} dimension in the keys")

            column = _dispatch.unique(new_keys.column(f"k_{n - 2}"))
            if not _dispatch.max(column) < lambda_max:
                raise ValueError(
                    f"expected only all values for k_{n - 2} to be below {lambda_max}"
                )

            l_keys.insert(0, f"k_{n - 2}")

        l_keys.insert(0, f"l_{n}")

    # de-couple the basis back into a product of L=1 spherical harmonics
    if cg_coefficients is None:
        if torch_jit_is_scripting():
            raise ValueError(
                "in TorchScript mode, `cg_coefficients` must be pre-computed "
                "and given to this function explicitly"
            )
        else:
            cg_coefficients = _coefficients.calculate_cg_coefficients(
                lambda_max=lambda_max,
                cg_backend=cg_backend,
                # use_torch=use_torch,
                arrays_backend="torch" if use_torch else "numpy",
                dtype=tensor.block(0).values.dtype,
                device=tensor.block(0).values.device if use_torch else "cpu",
            )

    tensor_rank = lambda_max

    if new_component_names is None:
        new_component_names = [f"xyz_{i}" for i in range(1, tensor_rank + 1)]
    elif len(new_component_names) != tensor_rank:
        raise ValueError(
            f"`new_component_names` should have {tensor_rank} entries (the rank of the "
            f"tensor), got {len(new_component_names)}"
        )

    # work on a tensor that has the (possibly re-created) `l_{i}` dimensions in its keys
    tensor = TensorMap(new_keys, [block.copy() for block in tensor.blocks()])

    # position of the (single) `o3_mu` component that will be expanded into the
    # `tensor_rank` cartesian components
    mu_position = tensor.component_names.index("o3_mu")

    # Step 1: de-couple the basis back into a product of L=1 spherical harmonics.
    #
    # This reverses the iterative coupling done by `cartesian_to_spherical`: at each
    # step we split off the left-most L=1 spherical harmonic (`l_{m}`) from the coupled
    # `o3_lambda`, leaving the remaining (still coupled) intermediate to the right. The
    # intermediate is decoupled further on the next iteration, until only L=1 spherical
    # harmonics remain.
    component_position = mu_position
    m = tensor_rank
    while m >= 2:
        l_left_name = f"l_{m}"
        if m > 2:
            lambda_right_name = f"k_{m - 2}"
        else:
            lambda_right_name = "l_1"

        tensor = _do_decoupling(
            tensor=tensor,
            component_position=component_position,
            l_left_name=l_left_name,
            lambda_right_name=lambda_right_name,
            cg_coefficients=cg_coefficients,
            cg_backend=cg_backend,
        )
        component_position += 1
        m -= 1

    # Step 2: transform the `tensor_rank` L=1 spherical harmonics back into cartesian
    # components. Each L=1 component is stored as (m=-1, 0, 1), i.e. (y, z, x), so a
    # `roll` by +1 brings it back to (x, y, z). The reconstructed components occupy the
    # axes `mu_position, ..., mu_position + tensor_rank - 1`.
    new_blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        axes = [mu_position + 1 + i for i in range(tensor_rank)]
        values = _dispatch.roll(
            block.values,
            shifts=[1] * tensor_rank,
            axis=axes,
        )

        new_components: List[Labels] = []
        for idx, component in enumerate(block.components):
            if mu_position <= idx < mu_position + tensor_rank:
                new_components.append(
                    Labels(
                        names=[new_component_names[idx - mu_position]],
                        values=_dispatch.int_array_like(
                            [[0], [1], [2]], tensor.keys.values
                        ),
                    )
                )
            else:
                new_components.append(component)

        new_blocks.append(
            TensorBlock(values, block.samples, new_components, block.properties)
        )

    # remove the now-unused spherical keys
    new_keys = tensor.keys
    new_keys = new_keys.remove("o3_lambda")
    new_keys = new_keys.remove("o3_sigma")
    if "l_1" in new_keys.names:
        # only left over for rank 1 tensors (no decoupling happened)
        new_keys = new_keys.remove("l_1")

    return TensorMap(new_keys, new_blocks)


def _do_coupling(
    tensor: TensorMap,
    component_1: int,
    component_2: int,
    keep_l_in_keys: bool,
    iteration_index: int,
    cg_coefficients: TensorMap,
    cg_backend: str,
) -> TensorMap:
    """
    Go from an uncoupled product basis that behave like a product of spherical harmonics
    to a coupled basis that behaves like a single spherical harmonic.

    This function takes in a :py:class:`TensorMap` where two of the components
    (indicated by ``component_1`` and ``component_2``) behave like spherical harmonics
    ``Y^m1_l1`` and ``Y^m2_l2``, and project it onto a single spherical harmonic
    ``Y^M_L``. This transformation uses the following relation:

    ``|L M> = |l1 l2 L M> = \\sum_{m1 m2} <l1 m1 l2 m2|L M> |l1 m1> |l2 m2>``

    where ``<l1 m1 l2 m2|L M>`` are Clebsch-Gordan coefficients.

    The output will contain many blocks for each block in the input, matching all the
    different ``L`` (called ``o3_lambda`` in the code) required to do a full projection.

    This process can be iterated: a multi-dimensional array that is the product of many
    ``Y^m_l`` can be turned into a set of multiple terms transforming as a single
    ``Y^M_L``.

    :param tensor: input :py:class:`TensorMap`
    :param components_1: first component of the ``tensor`` behaving like spherical
        harmonics
    :param components_2: second component of the ``tensor`` behaving like spherical
        harmonics
    :param keep_l_in_keys: whether ``l1`` and ``l2`` (the original spherical harmonic
        degrees) should be kept in the keys. This can be useful to undo this
        transformation (or even required if there is more than one path to get to a
        given value for ``o3_lambda``)
    :param iteration_index: when iterating the coupling, this should be the number of
        iterations already done (i.e. the number of time this function has been called)
    :param cg_coefficients: pre-computed set of Clebsch-Gordan coefficients
    :param cg_backend: which backend to use for the calculations

    :return: :py:class:`TensorMap` using the coupled basis. This will contain the same
        keys as the input ``tensor``, plus ``o3_lambda``. The components in positions
        ``components_1`` and ``components_2`` will be replaced by a single ``o3_mu``
        component.
    """
    assert component_2 == component_1 + 1

    new_keys = tensor.keys

    if "o3_lambda" in tensor.keys.names:
        old_sigmas = new_keys.column("o3_sigma")
        new_keys = new_keys.remove("o3_sigma")
    else:
        old_sigmas = _dispatch.int_array_like([1] * len(new_keys), new_keys.values)

    if keep_l_in_keys:
        array_of_ones = _dispatch.int_array_like([1] * len(new_keys), new_keys.values)
        if "o3_lambda" in tensor.keys.names:
            assert iteration_index > 0
            new_keys = new_keys.rename("o3_lambda", f"k_{iteration_index}")
            new_keys = new_keys.insert(0, f"l_{iteration_index + 2}", array_of_ones)
        else:
            assert iteration_index == 0
            new_keys = new_keys.insert(0, "l_1", array_of_ones)
            new_keys = new_keys.insert(0, "l_2", array_of_ones)

    new_keys_values: List[List[int]] = []
    new_blocks: List[TensorBlock] = []
    for key_idx, block in enumerate(tensor.blocks()):
        key = new_keys.entry(key_idx)
        old_sigma = int(old_sigmas[key_idx])

        # get l1, l2 from the block's shape
        block_shape = block.values.shape
        l1 = (block_shape[component_1 + 1] - 1) // 2
        l2 = (block_shape[component_2 + 1] - 1) // 2

        # reshape the values to look like (n_s, 2*l1 + 1, 2*l2 + 1, n_p)
        shape_before = 1
        for axis in range(component_1 + 1):
            shape_before *= block_shape[axis]

        shape_after = 1
        for axis in range(component_2 + 2, len(block_shape)):
            shape_after *= block_shape[axis]

        array = block.values.reshape(
            shape_before,
            block.values.shape[component_1 + 1],
            block.values.shape[component_2 + 1],
            shape_after,
        )

        # generate the set of o3_lambda to compute
        o3_lambdas = list(range(max(l1, l2) - min(l1, l2), (l1 + l2) + 1))

        # actual calculation
        outputs = _coefficients.cg_couple(
            array, o3_lambdas, cg_coefficients, cg_backend
        )

        # create one block for each output of `cg_couple`
        for o3_lambda, values in zip(o3_lambdas, outputs, strict=True):
            o3_sigma = int(old_sigma * (-1) ** (l1 + l2 + o3_lambda))
            new_keys_values.append(
                [o3_lambda, o3_sigma] + _dispatch.to_int_list(key.values)
            )

            new_shape = list(block.values.shape)
            new_shape.pop(component_2 + 1)
            new_shape[component_1 + 1] = 2 * o3_lambda + 1

            new_components = block.components
            new_components.pop(component_2)
            new_components[component_1] = Labels(
                "o3_mu",
                _dispatch.int_array_like(
                    [[mu] for mu in range(-o3_lambda, o3_lambda + 1)], new_keys.values
                ),
            )

            new_blocks.append(
                TensorBlock(
                    values.reshape(new_shape),
                    samples=block.samples,
                    components=new_components,
                    properties=block.properties,
                )
            )

    new_keys = Labels(
        ["o3_lambda", "o3_sigma"] + new_keys.names,
        _dispatch.int_array_like(new_keys_values, new_keys.values),
    )
    return TensorMap(new_keys, new_blocks)


def _do_decoupling(
    tensor: TensorMap,
    component_position: int,
    l_left_name: str,
    lambda_right_name: str,
    cg_coefficients: TensorMap,
    cg_backend: str,
) -> TensorMap:
    """
    Undo a single coupling step performed by :py:func:`_do_coupling`.

    The component at ``component_position`` (which transforms like a single spherical
    harmonic ``Y^M_L`` with ``L = o3_lambda``) is split back into a product of two
    spherical harmonics: an ``L=1`` term (given by ``l_left_name``, always equal to 1)
    at ``component_position`` and an intermediate term (given by ``lambda_right_name``)
    at ``component_position + 1``. This is the inverse of coupling those two terms into
    ``o3_lambda`` via Clebsch-Gordan coefficients.

    All blocks that share the same keys except for ``o3_lambda`` (and the dependent
    ``o3_sigma``) are grouped together and summed through :py:func:`cg_uncouple`, so the
    multiple ``o3_lambda`` blocks created by the forward coupling collapse back into a
    single block.

    :param tensor: input :py:class:`TensorMap`, in coupled form
    :param component_position: index (in ``block.components``) of the component to split
    :param l_left_name: name of the key holding the degree of the left-hand spherical
        harmonic to recover (``l_{m}``); its value is always 1
    :param lambda_right_name: name of the key holding the degree of the intermediate
        (right-hand) spherical harmonic to recover (``k_{m - 2}``, or ``l_1`` for the
        last decoupling step)
    :param cg_coefficients: pre-computed Clebsch-Gordan coefficients
    :param cg_backend: which backend to use for the calculation

    :return: :py:class:`TensorMap` with ``o3_lambda`` set to the recovered intermediate
        degree, ``l_left_name`` and ``lambda_right_name`` removed from the keys, and the
        single component at ``component_position`` replaced by two components.
    """
    keys = tensor.keys
    names = keys.names
    blocks = tensor.blocks()

    lambda_values = _dispatch.to_int_list(keys.column("o3_lambda"))
    sigma_values = _dispatch.to_int_list(keys.column("o3_sigma"))
    left_values = _dispatch.to_int_list(keys.column(l_left_name))
    right_values = _dispatch.to_int_list(keys.column(lambda_right_name))

    # group blocks by all key dimensions except `o3_lambda` and `o3_sigma`
    group_names: List[str] = []
    for name in names:
        if name != "o3_lambda" and name != "o3_sigma":
            group_names.append(name)
    group_columns = [_dispatch.to_int_list(keys.column(name)) for name in group_names]

    # output keys: keep everything except the two dimensions we are recovering, and put
    # back `o3_lambda` and `o3_sigma` at the front
    kept_names: List[str] = []
    for name in group_names:
        if name != l_left_name and name != lambda_right_name:
            kept_names.append(name)
    out_names = ["o3_lambda", "o3_sigma"] + kept_names

    group_signatures: List[List[int]] = []
    group_members: List[List[int]] = []
    for block_idx in range(len(blocks)):
        signature = [group_columns[c][block_idx] for c in range(len(group_names))]
        found = -1
        for g in range(len(group_signatures)):
            if group_signatures[g] == signature:
                found = g
                break
        if found == -1:
            group_signatures.append(signature)
            group_members.append([block_idx])
        else:
            group_members[found].append(block_idx)

    new_keys_values: List[List[int]] = []
    new_blocks: List[TensorBlock] = []
    mu_axis = component_position + 1
    for g in range(len(group_signatures)):
        members = group_members[g]
        first = members[0]

        l1 = left_values[first]  # always 1
        l2 = right_values[first]

        ref_block = blocks[first]
        shape = ref_block.values.shape
        n_before = 1
        for axis in range(mu_axis):
            n_before *= shape[axis]
        n_after = 1
        for axis in range(mu_axis + 1, len(shape)):
            n_after *= shape[axis]

        o3_lambdas: List[int] = []
        arrays: List[Array] = []
        for block_idx in members:
            o3_lambda = lambda_values[block_idx]
            o3_lambdas.append(o3_lambda)
            arrays.append(
                blocks[block_idx].values.reshape(n_before, 2 * o3_lambda + 1, n_after)
            )

        uncoupled = _coefficients.cg_uncouple(
            arrays, l1, l2, o3_lambdas, cg_coefficients, cg_backend
        )

        new_shape = (
            list(shape[:mu_axis])
            + [2 * l1 + 1, 2 * l2 + 1]
            + list(shape[mu_axis + 1 :])
        )
        values = uncoupled.reshape(new_shape)

        new_components: List[Labels] = []
        for idx in range(len(ref_block.components)):
            if idx == component_position:
                new_components.append(
                    Labels(
                        names=[f"__sph_{component_position}"],
                        values=_dispatch.int_array_like(
                            [[mu] for mu in range(-l1, l1 + 1)], keys.values
                        ),
                    )
                )
                new_components.append(
                    Labels(
                        names=[f"__sph_{component_position + 1}"],
                        values=_dispatch.int_array_like(
                            [[mu] for mu in range(-l2, l2 + 1)], keys.values
                        ),
                    )
                )
            else:
                new_components.append(ref_block.components[idx])

        new_blocks.append(
            TensorBlock(
                values=values,
                samples=ref_block.samples,
                components=new_components,
                properties=ref_block.properties,
            )
        )

        # recover the `o3_sigma` of the intermediate. The forward coupling set
        # `o3_sigma = old_sigma * (-1)^(l1 + l2 + o3_lambda)`, and `(-1)^x` is its own
        # inverse, so we recover `old_sigma` the same way.
        old_sigma = sigma_values[first] * ((-1) ** (l1 + l2 + o3_lambdas[0]))

        signature = group_signatures[g]
        kept_values: List[int] = []
        for c in range(len(group_names)):
            if group_names[c] != l_left_name and group_names[c] != lambda_right_name:
                kept_values.append(signature[c])

        new_keys_values.append([l2, old_sigma] + kept_values)

    new_keys = Labels(
        out_names,
        _dispatch.int_array_like(new_keys_values, keys.values),
    )
    return TensorMap(new_keys, new_blocks)
