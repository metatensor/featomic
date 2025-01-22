import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap, operations

from featomic.clebsch_gordan import cartesian_to_spherical


@pytest.fixture
def cartesian():
    # the first block is completely symmetric
    values_1 = np.random.rand(10, 4, 3, 3, 3, 2)
    values_1[:, :, 0, 1, 0, :] = values_1[:, :, 0, 0, 1, :]
    values_1[:, :, 1, 0, 0, :] = values_1[:, :, 0, 0, 1, :]

    values_1[:, :, 0, 2, 0, :] = values_1[:, :, 0, 0, 2, :]
    values_1[:, :, 2, 0, 0, :] = values_1[:, :, 0, 0, 2, :]

    values_1[:, :, 1, 0, 1, :] = values_1[:, :, 0, 1, 1, :]
    values_1[:, :, 1, 1, 0, :] = values_1[:, :, 0, 1, 1, :]

    values_1[:, :, 2, 0, 2, :] = values_1[:, :, 0, 2, 2, :]
    values_1[:, :, 2, 2, 0, :] = values_1[:, :, 0, 2, 2, :]

    values_1[:, :, 2, 1, 2, :] = values_1[:, :, 2, 2, 1, :]
    values_1[:, :, 1, 2, 2, :] = values_1[:, :, 2, 2, 1, :]

    values_1[:, :, 1, 2, 1, :] = values_1[:, :, 1, 1, 2, :]
    values_1[:, :, 2, 1, 1, :] = values_1[:, :, 1, 1, 2, :]

    values_1[:, :, 0, 2, 1, :] = values_1[:, :, 0, 1, 2, :]
    values_1[:, :, 2, 0, 1, :] = values_1[:, :, 0, 1, 2, :]
    values_1[:, :, 1, 0, 2, :] = values_1[:, :, 0, 1, 2, :]
    values_1[:, :, 1, 2, 0, :] = values_1[:, :, 0, 1, 2, :]
    values_1[:, :, 2, 1, 0, :] = values_1[:, :, 0, 1, 2, :]

    block_1 = TensorBlock(
        values=values_1,
        samples=Labels.range("s", 10),
        components=[
            Labels.range("other", 4),
            Labels.range("xyz_1", 3),
            Labels.range("xyz_2", 3),
            Labels.range("xyz_3", 3),
        ],
        properties=Labels.range("p", 2),
    )

    # second block does not have any specific symmetry
    block_2 = TensorBlock(
        values=np.random.rand(12, 6, 3, 3, 3, 7),
        samples=Labels.range("s", 12),
        components=[
            Labels.range("other", 6),
            Labels.range("xyz_1", 3),
            Labels.range("xyz_2", 3),
            Labels.range("xyz_3", 3),
        ],
        properties=Labels.range("p", 7),
    )

    return TensorMap(Labels.range("key", 2), [block_1, block_2])


def test_cartesian_to_spherical_rank_1(cartesian):
    spherical = cartesian_to_spherical(cartesian, components=["xyz_1"])

    assert spherical.component_names == ["other", "o3_mu", "xyz_2", "xyz_3"]
    assert spherical.keys.names == ["o3_lambda", "o3_sigma", "key"]

    spherical = cartesian_to_spherical(
        cartesian,
        components=["xyz_1"],
        keep_l_in_keys=True,
    )
    assert spherical.keys.names == ["o3_lambda", "o3_sigma", "l_1", "key"]


def test_cartesian_to_spherical_rank_2(cartesian):
    spherical = cartesian_to_spherical(cartesian, components=["xyz_1", "xyz_2"])

    assert spherical.component_names == ["other", "o3_mu", "xyz_3"]
    assert spherical.keys == Labels(
        ["o3_lambda", "o3_sigma", "key"],
        np.array(
            [
                # only o3_sigma=1 in the symmetric block
                [0, 1, 0],
                [2, 1, 0],
                # all o3_sigma in the non-symmetric block
                [0, 1, 1],
                [1, -1, 1],
                [2, 1, 1],
            ]
        ),
    )

    spherical = cartesian_to_spherical(
        cartesian,
        components=["xyz_1", "xyz_2"],
        keep_l_in_keys=True,
        remove_blocks_threshold=None,
    )
    assert spherical.keys.names == ["o3_lambda", "o3_sigma", "l_2", "l_1", "key"]
    # all blocks are kept
    assert len(spherical.keys) == 6


def test_cartesian_to_spherical_rank_3(cartesian):
    spherical = cartesian_to_spherical(
        cartesian, components=["xyz_1", "xyz_2", "xyz_3"]
    )

    assert spherical.component_names == ["other", "o3_mu"]
    assert spherical.keys == Labels(
        ["o3_lambda", "o3_sigma", "l_3", "k_1", "l_2", "l_1", "key"],
        np.array(
            [
                # only o3_sigma=1 for the symmetric block, but there are multiple "path"
                # ("l_3", "k_1", "l_2", "l_1") that lead to o3_lambda=1
                [1, 1, 1, 0, 1, 1, 0],
                [1, 1, 1, 2, 1, 1, 0],
                [3, 1, 1, 2, 1, 1, 0],
                # all possible o3_sigma for the non-symmetric block
                [1, 1, 1, 0, 1, 1, 1],
                [0, -1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [2, -1, 1, 1, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 1],
                [2, -1, 1, 2, 1, 1, 1],
                [3, 1, 1, 2, 1, 1, 1],
            ]
        ),
    )

    spherical = cartesian_to_spherical(
        cartesian,
        components=["xyz_1", "xyz_2", "xyz_3"],
        remove_blocks_threshold=None,
    )
    # all blocks are kept, even those with norm=0
    assert len(spherical.keys) == 14


@pytest.mark.parametrize("cg_backend", ["python-dense", "python-sparse"])
@pytest.mark.parametrize(
    "components", [["xyz_1"], ["xyz_1", "xyz_12"], ["xyz_1", "xyz_2", "xyz_3"]]
)
def test_cartesian_to_spherical_and_back(cartesian, components, cg_backend):
    spherical = cartesian_to_spherical(
        cartesian,
        components=["xyz_1", "xyz_2", "xyz_3"],
        keep_l_in_keys=True,
        cg_backend=cg_backend,
    )

    assert "o3_lambda" in spherical.keys.names
    # TODO: check for identity after spherical_to_cartesian


def test_cartesian_to_spherical_errors(cartesian):
    message = "`components` should be a list, got <class 'str'>"
    with pytest.raises(TypeError, match=message):
        cartesian_to_spherical(cartesian, components="xyz_1")

    message = "'1' is not part of this tensor components"
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=[1, 2])

    message = "'not_there' is not part of this tensor components"
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=["not_there"])

    message = (
        "this function only supports consecutive components, "
        "\\['xyz_2', 'xyz_1'\\] are not"
    )
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=["xyz_2", "xyz_1"])

    message = (
        "this function only supports consecutive components, "
        "\\['xyz_1', 'xyz_3'\\] are not"
    )
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=["xyz_1", "xyz_3"])

    message = (
        "component 'other' in block for \\(key=0\\) should have \\[0, 1, 2\\] "
        "as values, got \\[0, 1, 2, 3\\] instead"
    )
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(cartesian, components=["other"])

    message = "`keep_l_in_keys` must be `True` for tensors of rank 3 and above"
    with pytest.raises(ValueError, match=message):
        cartesian_to_spherical(
            cartesian,
            components=["xyz_1", "xyz_2", "xyz_3"],
            keep_l_in_keys=False,
        )


def _l2_components_from_matrix(A):
    """
    *The* reference equations for projecting a cartesian rank 2 tensor to the
    irreducible spherical component with lambda = 2.
    """
    A = A.reshape(3, 3)

    l2_A = np.empty((5,))
    l2_A[0] = (A[0, 1] + A[1, 0]) / 2.0
    l2_A[1] = (A[1, 2] + A[2, 1]) / 2.0
    l2_A[2] = (2.0 * A[2, 2] - A[0, 0] - A[1, 1]) / ((2.0) * np.sqrt(3.0))
    l2_A[3] = (A[0, 2] + A[2, 0]) / 2.0
    l2_A[4] = (A[0, 0] - A[1, 1]) / 2.0

    return l2_A * np.sqrt(2)


def _l3_components_from_matrix(A):
    """
    *The* reference equations for projecting a cartesian rank 3 tensor to the
    irreducible spherical component with lambda = 3.
    """
    A = A.reshape(3, 3, 3)

    l3_A = np.empty((7,))

    # -3
    l3_A[0] = (A[0, 1, 0] + A[1, 0, 0] + A[0, 0, 1] - A[1, 1, 1]) / 2.0

    # -2
    l3_A[1] = (
        A[0, 1, 2] + A[1, 0, 2] + A[1, 2, 0] + A[2, 1, 0] + A[0, 2, 1] + A[2, 0, 1]
    ) / np.sqrt(6)

    # -1
    l3_A[2] = (
        4.0 * A[1, 2, 2]
        + 4.0 * A[2, 1, 2]
        + 4.0 * A[2, 2, 1]
        - 3.0 * A[1, 1, 1]
        - A[0, 0, 1]
        - A[0, 1, 0]
        - A[1, 0, 0]
    ) / np.sqrt(60)

    # 0
    l3_A[3] = (
        2.0 * A[2, 2, 2]
        - A[0, 2, 0]
        - A[2, 0, 0]
        - A[0, 0, 2]
        - A[1, 2, 1]
        - A[2, 1, 1]
        - A[1, 1, 2]
    ) / np.sqrt(10)

    # 1
    l3_A[4] = (
        4.0 * A[0, 2, 2]
        + 4.0 * A[2, 0, 2]
        + 4.0 * A[2, 2, 0]
        - 3.0 * A[0, 0, 0]
        - A[1, 1, 0]
        - A[0, 1, 1]
        - A[1, 0, 1]
    ) / np.sqrt(60)

    # 2
    l3_A[5] = (
        A[0, 0, 2] + A[0, 2, 0] + A[2, 0, 0] - A[1, 1, 2] - A[1, 2, 1] - A[2, 1, 1]
    ) / np.sqrt(6)

    # 3
    l3_A[6] = (A[0, 0, 0] - A[1, 1, 0] - A[0, 1, 1] - A[1, 0, 1]) / 2.0

    return l3_A


@pytest.mark.parametrize("cg_backend", ["python-dense", "python-sparse"])
def test_cartesian_to_spherical_rank_2_by_equation(cg_backend):
    """
    Tests cartesian_to_spherical for a random rank-2 tensor by comparing the result to
    the result from *the* reference equation.
    """
    # Build the reference lambda = 2 component
    random_rank_2_arr = np.random.rand(100, 3, 3, 1)
    l2_reference = TensorMap(
        keys=Labels(["o3_lambda", "o3_sigma"], np.array([[2, 1]])),
        blocks=[
            TensorBlock(
                samples=Labels(["system"], np.arange(100).reshape(-1, 1)),
                components=[Labels(["o3_mu"], np.arange(-2, 3).reshape(-1, 1))],
                properties=Labels(["_"], np.array([[0]])),
                values=np.stack(
                    [_l2_components_from_matrix(A[..., 0]) for A in random_rank_2_arr]
                ).reshape(random_rank_2_arr.shape[0], 5, 1),
            )
        ],
    )

    # Build the cartesian tensor and do cartesian to spherical
    rank_2_input_cart = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=random_rank_2_arr,
                samples=Labels(["system"], np.arange(100).reshape(-1, 1)),
                components=[
                    Labels([a], np.arange(3).reshape(-1, 1)) for a in ["xyz1", "xyz2"]
                ],
                properties=Labels(["_"], np.array([[0]])),
            )
        ],
    )
    rank_2_input_sph = cartesian_to_spherical(
        rank_2_input_cart, ["xyz1", "xyz2"], cg_backend=cg_backend
    )

    # Extract the lambda = 2 component
    l2_input = operations.drop_blocks(
        operations.remove_dimension(rank_2_input_sph, "keys", "_"),
        keys=Labels(["o3_lambda"], np.array([[0], [1]])),
    )

    assert operations.equal_metadata(l2_input, l2_reference)
    assert operations.allclose(l2_input, l2_reference)


@pytest.mark.parametrize("cg_backend", ["python-dense", "python-sparse"])
def test_cartesian_to_spherical_rank_3_by_equation(cg_backend):
    """
    Tests cartesian_to_spherical for a random rank-2 tensor by comparing the result to
    the result from *the* reference equation.
    """
    # Build the reference lambda = 2 component
    random_rank_3_arr = np.random.rand(100, 3, 3, 3, 1)
    l3_reference = TensorMap(
        keys=Labels(["o3_lambda", "o3_sigma"], np.array([[3, 1]])),
        blocks=[
            TensorBlock(
                samples=Labels(["system"], np.arange(100).reshape(-1, 1)),
                components=[Labels(["o3_mu"], np.arange(-3, 4).reshape(-1, 1))],
                properties=Labels(["_"], np.array([[0]])),
                values=np.stack(
                    [_l3_components_from_matrix(A[..., 0]) for A in random_rank_3_arr]
                ).reshape(random_rank_3_arr.shape[0], 7, 1),
            )
        ],
    )

    # Build the cartesian tensor and do cartesian to spherical
    rank_3_input_cart = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=random_rank_3_arr,
                samples=Labels(["system"], np.arange(100).reshape(-1, 1)),
                components=[
                    Labels([a], np.arange(3).reshape(-1, 1))
                    for a in ["xyz1", "xyz2", "xyz3"]
                ],
                properties=Labels(["_"], np.array([[0]])),
            )
        ],
    )
    rank_3_input_sph = cartesian_to_spherical(
        rank_3_input_cart, ["xyz1", "xyz2", "xyz3"], cg_backend=cg_backend
    )

    # Extract the lambda = 3 component
    l3_input = operations.drop_blocks(
        operations.remove_dimension(rank_3_input_sph, "keys", "_"),
        keys=Labels(["o3_lambda"], np.array([[0], [1], [2]])),
    )
    for dim in ["l_3", "k_1", "l_2", "l_1"]:
        l3_input = operations.remove_dimension(l3_input, "keys", dim)

    assert operations.equal_metadata(l3_input, l3_reference)
    assert operations.allclose(l3_input, l3_reference)
