import ase
import numpy as np
import pytest
import torch

from featomic.torch import systems_to_torch


def test_system_conversion_from_ase():
    atoms = ase.Atoms(
        "CO",
        positions=[(0, 0, 0), (0, 0, 2)],
        cell=4 * np.eye(3),
        pbc=[True, True, True],
    )

    system = systems_to_torch(atoms)

    assert isinstance(system, torch.ScriptObject)

    assert isinstance(system.types, torch.Tensor)
    assert torch.all(system.types == torch.tensor([6, 8]))
    assert system.types.dtype == torch.int32
    assert not system.types.requires_grad

    assert isinstance(system.positions, torch.Tensor)
    assert torch.all(system.positions == torch.tensor([(0, 0, 0), (0, 0, 2)]))
    assert system.positions.dtype == torch.float64
    assert not system.positions.requires_grad

    assert isinstance(system.cell, torch.Tensor)
    assert torch.all(system.cell == 4 * torch.eye(3))
    assert system.cell.dtype == torch.float64
    assert not system.cell.requires_grad

    system = systems_to_torch(atoms, positions_requires_grad=True)

    assert system.positions.requires_grad
    assert not system.cell.requires_grad

    # we can send a torch System through this function, and change the requires_grad
    system = systems_to_torch(
        system,
        cell_requires_grad=True,
        positions_requires_grad=False,
    )

    assert not system.positions.requires_grad
    assert system.cell.requires_grad

    # test a list of ase.Atoms
    systems = systems_to_torch([atoms, atoms])
    assert isinstance(systems[0], torch.ScriptObject)


def test_system_conversion_non_periodic():
    """Test conversion of non-periodic systems."""
    atoms = ase.Atoms(
        "CO",
        positions=[(0, 0, 0), (0, 0, 2)],
        cell=None,
        pbc=[False, False, False],
    )

    system = systems_to_torch(atoms)

    assert isinstance(system, torch.ScriptObject)
    assert torch.all(system.cell == 0)

    # test a list of non-periodic systems
    systems = systems_to_torch([atoms, atoms])
    assert isinstance(systems[0], torch.ScriptObject)


def test_system_conversion_non_periodic_with_cell_warning():
    """Test warning when converting non-periodic system with non-zero cell."""
    atoms = ase.Atoms(
        "CO",
        positions=[(0, 0, 0), (0, 0, 2)],
        cell=4 * np.eye(3),
        pbc=[False, False, False],
    )

    message = (
        "A conversion to `System` was requested for a system with non-zero "
        "cell vectors but where all periodic boundary conditions are "
        "disabled. The cell vectors will be set to zero."
    )
    with pytest.warns(UserWarning, match=message):
        system = systems_to_torch(atoms)

    assert torch.all(system.cell == 0)


def test_system_conversion_mixed_pbc_warning():
    """Test warning when converting system with mixed PBC."""
    atoms = ase.Atoms(
        "CO",
        positions=[(0, 0, 0), (0, 0, 2)],
        cell=4 * np.eye(3),
        pbc=[False, False, True],
    )

    message = (
        "A conversion to `System` was requested with mixed periodic boundary "
        "conditions. Only fully periodic or fully non-periodic systems are "
        "fully supported. Setting non-periodic cell vectors to zero."
    )
    with pytest.warns(UserWarning, match=message):
        system = systems_to_torch(atoms)

    # Non-periodic directions should be zeroed
    assert torch.all(system.cell[0] == 0)
    assert torch.all(system.cell[1] == 0)
    # Periodic direction should be preserved
    assert torch.all(system.cell[2] != 0)
