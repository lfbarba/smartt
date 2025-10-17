import torch

from pyTT import generate_spherical_harmonic_lobes


def test_generate_spherical_harmonic_lobes_shapes():
    lobes = generate_spherical_harmonic_lobes(l=2, m=1, theta_resolution=24, phi_resolution=48)

    assert lobes.theta.shape == (24, 48)
    assert lobes.phi.shape == (24, 48)
    assert lobes.values.shape == (24, 48)
    assert lobes.radius.shape == (24, 48)
    assert lobes.x.shape == (24, 48)
    assert lobes.y.shape == (24, 48)
    assert lobes.z.shape == (24, 48)

    # Basic sanity checks
    assert torch.isfinite(lobes.values).all()
    assert torch.isfinite(lobes.x).all()
    # Ensure deformation is applied (not a perfect sphere)
    assert torch.max(torch.abs(lobes.radius - lobes.radius.mean())) > 0
