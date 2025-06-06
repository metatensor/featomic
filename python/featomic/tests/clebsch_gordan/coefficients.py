import numpy as np
import pytest

from featomic.clebsch_gordan._coefficients import _complex2real


scipy = pytest.importorskip("scipy")


def complex_to_real_manual(sph):
    # following https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    ell = (sph.shape[1] - 1) // 2

    real = np.zeros(sph.shape)
    for m in range(-ell, ell + 1):
        if m < 0:
            real[:, ell + m] = np.sqrt(2) * (-1) ** m * np.imag(sph[:, ell + abs(m)])
        elif m == 0:
            assert np.all(np.imag(sph[:, ell + m]) == 0)
            real[:, ell + m] = np.real(sph[:, ell + m])
        else:
            real[:, ell + m] = np.sqrt(2) * (-1) ** m * np.real(sph[:, ell + m])

    return real


def complex_to_real_matrix(sph):
    ell = (sph.shape[1] - 1) // 2

    matrix = _complex2real(ell, sph)

    real = matrix @ sph.T
    real = real.T

    assert np.allclose(np.imag(real), 0)
    return np.real(real)


def sph_harm(ell, m, theta, phi):
    if tuple(map(int, scipy.__version__.split("."))) > (1, 15, 0):
        return scipy.special.sph_harm_y(ell, m, theta, phi)
    else:
        return scipy.special.sph_harm(m, ell, theta, phi)


def test_complex_to_real():
    theta = 2 * np.pi * np.random.rand(10)
    phi = np.pi * np.random.rand(10)

    for ell in range(4):
        values = np.zeros((10, 2 * ell + 1), dtype=np.complex128)
        for m in range(-ell, ell + 1):
            values[:, ell + m] = sph_harm(ell, m, theta, phi)

        real_manual = complex_to_real_manual(values)
        real_matrix = complex_to_real_matrix(values)

        assert np.allclose(real_manual, real_matrix)
