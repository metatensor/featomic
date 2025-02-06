import metatensor
import numpy as np
import pytest
from metatensor import Labels
from numpy.testing import assert_equal

from featomic import SphericalExpansion, SphericalExpansionByPair
from featomic.clebsch_gordan import (
    EquivariantPowerSpectrum,
    EquivariantPowerSpectrumByPair,
)


# Try to import some modules
ase = pytest.importorskip("ase")
import ase.io  # noqa: E402, F811


SPHEX_HYPERS_SMALL = {
    "cutoff": {
        "radius": 2.5,
        "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.2,
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": 2,
        "radial": {"type": "Gto", "max_radial": 1},
    },
}

SPHEX_HYPERS_LARGE = {
    "cutoff": {
        "radius": 5.0,
        "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.2,
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": 2,
        "radial": {"type": "Gto", "max_radial": 1},
    },
}
# ============ Helper functions ============


def h2o_periodic():
    return [
        ase.Atoms(
            symbols=["O", "H", "H"],
            positions=[
                [2.56633400, 2.50000000, 2.50370100],
                [1.97361700, 1.73067300, 2.47063400],
                [1.97361700, 3.26932700, 2.47063400],
            ],
            cell=[3, 3, 3],
            pbc=[True, True, True],
        )
    ]


# =========== Test EquivariantPowerSpectrumByPair vs EquivariantPowerSpectrum ==========


def test_equivariant_power_spectrum_vs_equivariant_power_spectrum_by_pair():
    """
    Tests for exact equivalence between EquivariantPowerSpectrumByPair and
    EquivariantPowerSpectrum after metadata manipulation and reduction over samples.
    """

    # Build an EquivariantPowerSpectrum
    ps = EquivariantPowerSpectrum(SphericalExpansion(**SPHEX_HYPERS_SMALL)).compute(
        h2o_periodic(),
        selected_keys=Labels(names=["o3_lambda"], values=np.array([0]).reshape(-1, 1)),
        neighbors_to_properties=False,
    )

    # Build an EquivariantPowerSpectrumByPair
    ps_by_pair = EquivariantPowerSpectrumByPair(
        SphericalExpansion(**SPHEX_HYPERS_SMALL),
        SphericalExpansionByPair(**SPHEX_HYPERS_SMALL),
    ).compute(
        h2o_periodic(),
        selected_keys=Labels(names=["o3_lambda"], values=np.array([0]).reshape(-1, 1)),
        neighbors_to_properties=False,
    )

    # Manipulate metadata to match
    reduced_ps_by_pair = metatensor.rename_dimension(
        ps_by_pair, "keys", "first_atom_type", "center_type"
    )
    reduced_ps_by_pair = metatensor.rename_dimension(
        reduced_ps_by_pair,
        "keys",
        "second_atom_type",
        "neighbor_2_type",
    )
    reduced_ps_by_pair = metatensor.rename_dimension(
        reduced_ps_by_pair,
        "samples",
        "first_atom",
        "atom",
    )
    # Sum over `second_atom` and `cell_shift` samples
    reduced_ps_by_pair = metatensor.sum_over_samples(
        reduced_ps_by_pair,
        ["second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
    )

    for k, b1 in ps.items():
        b2 = reduced_ps_by_pair.block(k)
        if b1.values.shape != b2.values.shape:
            assert b2.values.shape[0] == 0
            assert np.allclose(np.linalg.norm(b1.values), np.linalg.norm(b2.values))
        else:
            assert np.allclose(b1.values, b2.values)


def test_sample_selection() -> None:
    """Tests that the sample selection works as expected.
    By first computing the equivariant power spectrum by pair for all atoms in H2O.
    Then first for atom 1 and then atom 2  and 3.
    Their join should be identical to computing it for all atoms.
    """

    frame = h2o_periodic()

    powspec_by_pair_calc = EquivariantPowerSpectrumByPair(
        SphericalExpansion(**SPHEX_HYPERS_SMALL),
        SphericalExpansionByPair(**SPHEX_HYPERS_SMALL),
    )

    label_1st = metatensor.Labels(
        ["system", "atom"], np.array([[0, 0]], dtype=np.int32)
    )

    label_2nd = metatensor.Labels(
        ["system", "atom"], np.array([[0, 1], [0, 2]], dtype=np.int32)
    )

    powspec_1 = powspec_by_pair_calc.compute(
        frame, neighbors_to_properties=True, selected_samples=label_1st
    )

    powspec_2 = powspec_by_pair_calc.compute(
        frame, neighbors_to_properties=True, selected_samples=label_2nd
    )

    powspec_3 = metatensor.join(
        [powspec_1, powspec_2], axis="samples", remove_tensor_name=True
    )
    powspec_4 = powspec_by_pair_calc.compute(frame, neighbors_to_properties=True)

    assert metatensor.equal(powspec_3, powspec_4)
    assert not metatensor.equal(powspec_2, powspec_4)
    assert not metatensor.equal(powspec_1, powspec_4)


def test_equivariant_power_spectrum_neighbors_to_properties():
    """
    Tests that computing an EquivariantPowerSpectrumByPair is equivalent when passing
    `neighbors_to_properties` as both True and False (after metadata manipulation).
    """
    # Build an EquivariantPowerSpectrum
    powspec_calc = EquivariantPowerSpectrumByPair(
        SphericalExpansion(**SPHEX_HYPERS_LARGE),
        SphericalExpansionByPair(**SPHEX_HYPERS_LARGE),
    )

    # Compute the first. Move keys after CG step
    powspec_1 = powspec_calc.compute(
        h2o_periodic(),
        neighbors_to_properties=False,
    )
    powspec_1 = powspec_1.keys_to_properties(["neighbor_1_type"])

    # Compute the second.  Move keys before the CG step
    powspec_2 = powspec_calc.compute(
        h2o_periodic(),
        neighbors_to_properties=True,
    )

    # Permute properties dimensions to match ``powspec_1`` and sort
    powspec_2 = metatensor.sort(
        metatensor.permute_dimensions(powspec_2, "properties", [2, 0, 1, 3, 4])
    )

    # Check equivalent
    powspec_1 = metatensor.sort(powspec_1)
    metatensor.equal_metadata_raise(powspec_1, powspec_2)
    metatensor.allclose_raise(powspec_1, powspec_2)


def test_fill_types_option() -> None:
    """
    Test that ``neighbor_types`` options adds arbitrary atomic neighbor types.
    """

    frames = [
        ase.Atoms("H", positions=np.zeros([1, 3])),
        ase.Atoms("O", positions=np.zeros([1, 3])),
    ]

    neighbor_types = [1, 8, 10]
    calculator = EquivariantPowerSpectrumByPair(
        calculator_1=SphericalExpansion(**SPHEX_HYPERS_SMALL),
        calculator_2=SphericalExpansionByPair(**SPHEX_HYPERS_SMALL),
        neighbor_types=neighbor_types,
    )

    descriptor = calculator.compute(frames, neighbors_to_properties=True)

    print(descriptor[0].properties["neighbor_1_type"])

    assert_equal(np.unique(descriptor[0].properties["neighbor_1_type"]), neighbor_types)
