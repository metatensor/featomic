import numpy as np
from metatensor import Labels

from featomic import SphericalExpansion

from ..test_systems import SystemForTests


def test_gradients_sample_selection():
    system = SystemForTests()

    hypers = {
        "cutoff": {"radius": 3.0, "smoothing": {"type": "ShiftedCosine", "width": 0.1}},
        "density": {"type": "Gaussian", "width": 0.3},
        "basis": {
            "type": "TensorProduct",
            "max_angular": 2,
            "radial": {"type": "Gto", "max_radial": 6},
        },
    }

    calculator = SphericalExpansion(**hypers)

    # Calculate for the full system
    spx_full = calculator.compute(system, gradients=["positions"])
    spx_full = spx_full.keys_to_properties("neighbor_type")
    spx_full = spx_full.keys_to_samples("center_type")

    # Calculate for the selected subset
    samples_selection = Labels(names=["atom"], values=np.array([[3], [0]]))

    spx_subset = calculator.compute(
        system,
        selected_samples=samples_selection,
        gradients=["positions"],
    )
    spx_subset = spx_subset.keys_to_properties("neighbor_type")
    spx_subset = spx_subset.keys_to_samples("center_type")

    # Compare gradients
    for key in spx_subset.keys:
        block_full = spx_full.block(key)
        block_subset = spx_subset.block(key)

        grad_full = block_full.gradient("positions")
        grad_subset = block_subset.gradient("positions")

        # For each sample in the subset, find the corresponding sample in the full
        # calculation and compare the gradients
        for sample_i, sample in enumerate(block_subset.samples):
            # Find index in full samples
            full_idx = block_full.samples.position(sample)
            assert full_idx is not None

            # Get gradients for this sample
            subset_grad_mask = grad_subset.samples["sample"] == sample_i
            full_grad_mask = grad_full.samples["sample"] == full_idx

            subset_grads = grad_subset.values[subset_grad_mask]
            full_grads = grad_full.values[full_grad_mask]

            # The ordering of gradient samples (neighbors) might differ, so we need to
            # match them
            subset_grad_samples = grad_subset.samples.column("atom")[subset_grad_mask]
            full_grad_samples = grad_full.samples.column("atom")[full_grad_mask]

            for grad_i, neighbor_i in enumerate(subset_grad_samples):
                # Find corresponding neighbor in full gradients
                # We look for the entry with the same neighbor atom index
                match_mask = full_grad_samples == neighbor_i
                assert np.sum(match_mask) == 1

                subset_val = subset_grads[grad_i]
                full_val = full_grads[match_mask][0]

                np.testing.assert_allclose(
                    subset_val,
                    full_val,
                    err_msg=f"Mismatch for center {sample[1]}, neighbor {neighbor_i}",
                )
