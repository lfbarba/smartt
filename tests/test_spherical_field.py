import math

import torch
from e3nn import o3

from pyTT import SphericalBesselHarmonicField


def _default_dtype() -> torch.dtype:
    return torch.get_default_dtype()


def _device() -> torch.device:
    return torch.device("cpu")


def test_forward_matches_manual_construction():
    field = SphericalBesselHarmonicField(
        dims=(1, 1, 1),
        max_l=0,
        num_radial=1,
        learn_k=False,
        k_init=math.pi,
        radius=1.0,
    )

    with torch.no_grad():
        field.coeffs[0].zero_()
        field.coeffs[0][0, 0, 0] = 1.0

    sample_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.1, 0.3],
            [0.5, -0.4, 0.1],
        ],
        dtype=_default_dtype(),
        device=_device(),
    ).unsqueeze(0)

    values = field(sample_points)

    r = torch.norm(sample_points, dim=-1)
    directions = torch.zeros_like(sample_points)
    non_zero = r > 0
    directions[non_zero] = sample_points[non_zero] / r[non_zero].unsqueeze(-1)
    directions[~non_zero] = torch.tensor([0.0, 0.0, 1.0], dtype=_default_dtype())

    k_value = field._k_values_for_l(0)[0]
    x = r * k_value
    radial = torch.where(
        x == 0,
        torch.ones_like(x),
        torch.sin(x) / x,
    )
    harmonics = o3.spherical_harmonics([0], directions, normalize=True, normalization="component")
    expected = radial * harmonics.squeeze(-1)

    assert torch.allclose(values, expected, atol=1e-6)


def test_slice_respects_mask_and_matches_forward():
    field = SphericalBesselHarmonicField(
        dims=(2, 1, 1),
        max_l=0,
        num_radial=1,
        learn_k=False,
        k_init=math.pi,
        radius=1.0,
    )

    with torch.no_grad():
        field.coeffs[0].zero_()
        field.coeffs[0][0, 0, 0] = 1.0
        field.coeffs[0][1, 0, 0] = 2.0

    normal = torch.tensor([0.0, 0.0, 1.0], dtype=_default_dtype())
    mask = torch.tensor([True, False], dtype=torch.bool).reshape(2, 1, 1)
    resolution = 24

    slice_values = field.slice(normal, mask=mask, grid_resolution=resolution)

    assert slice_values.shape == (2, 1, 1, resolution, resolution)
    assert torch.allclose(slice_values[1], torch.zeros_like(slice_values[1]))

    lin = torch.linspace(-field.radius, field.radius, resolution, dtype=_default_dtype())
    xx, yy = torch.meshgrid(lin, lin, indexing="ij")
    plane_points = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1).reshape(1, resolution * resolution, 3)

    forward_values = field(
        plane_points,
        selection=torch.tensor([0], dtype=torch.int64),
    ).reshape(1, resolution, resolution)

    radial_mask = (xx.pow(2) + yy.pow(2)) <= field.radius ** 2
    expected = forward_values.clone()
    expected[0][~radial_mask] = 0.0

    assert torch.allclose(slice_values[0, 0, 0], expected[0], atol=1e-5)


def test_slice_flatten_returns_active_only():
    field = SphericalBesselHarmonicField(
        dims=(3, 1, 1),
        max_l=0,
        num_radial=1,
        learn_k=False,
        k_init=math.pi,
        radius=1.0,
    )

    with torch.no_grad():
        field.coeffs[0].zero_()
        field.coeffs[0][0, 0, 0] = 0.5
        field.coeffs[0][1, 0, 0] = 1.0
        field.coeffs[0][2, 0, 0] = 1.5

    normal = torch.tensor([0.0, 0.0, 1.0], dtype=_default_dtype())
    mask = torch.tensor([True, False, True], dtype=torch.bool).reshape(3, 1, 1)
    resolution = 16

    flattened = field.slice(normal, mask=mask, grid_resolution=resolution, flatten=True)
    full = field.slice(normal, mask=mask, grid_resolution=resolution, flatten=False)
    full_all = field.slice(normal, mask=None, grid_resolution=resolution, flatten=True)

    assert flattened.shape == (2, resolution, resolution)
    assert full.shape == (3, 1, 1, resolution, resolution)
    assert full_all.shape == (3, resolution, resolution)

    assert torch.allclose(flattened[0], full[0, 0, 0])
    assert torch.allclose(flattened[1], full[2, 0, 0])


def test_shell_harmonics_matches_manual_coefficients():
    field = SphericalBesselHarmonicField(
        dims=(2, 1, 1),
        max_l=1,
        num_radial=2,
        learn_k=False,
        k_init=[math.pi, 2.0 * math.pi],
        radius=1.0,
    )

    with torch.no_grad():
        # ℓ=0 has dim 1, ℓ=1 dim 3
        field.coeffs[0][0] = torch.tensor([[0.5], [0.2]])
        field.coeffs[0][1] = torch.tensor([[1.0], [0.1]])
        field.coeffs[1][0] = torch.tensor(
            [[0.1, -0.2, 0.3],
             [0.05, 0.15, -0.25]],
        )
        field.coeffs[1][1] = torch.tensor(
            [[-0.3, 0.2, 0.1],
             [0.4, -0.1, 0.05]],
        )

    radii = torch.tensor([0.3, 0.8], dtype=_default_dtype())
    expansion = field.shell_harmonics(radii)

    assert expansion.irreps == o3.Irreps("0e + 1o")
    assert expansion.data.shape == (2, 1, 1, radii.numel(), expansion.irreps.dim)

    for sphere in range(2):
        expected = torch.zeros((radii.numel(), expansion.irreps.dim), dtype=_default_dtype())
        offset = 0
        for idx, l in enumerate(field._l_values):
            dim = 2 * l + 1
            k_vals = field._k_values_for_l(idx)
            coeff = field.coeffs[idx][sphere]
            for radius_index, radius_value in enumerate(radii):
                jl = field._spherical_jn(l, radius_value * k_vals)
                expected[radius_index, offset:offset + dim] = torch.matmul(jl, coeff)
            offset += dim

        actual = expansion.data[sphere, 0, 0]
        assert torch.allclose(actual, expected, atol=1e-6)


def test_shell_harmonics_respects_mask_and_flatten():
    field = SphericalBesselHarmonicField(
        dims=(3, 1, 1),
        max_l=0,
        num_radial=1,
        learn_k=False,
        k_init=math.pi,
        radius=1.0,
    )

    with torch.no_grad():
        field.coeffs[0][:, 0, 0] = torch.tensor([0.1, 0.2, 0.3])

    radii = [0.25]
    mask = torch.tensor([True, False, True], dtype=torch.bool).reshape(3, 1, 1)

    flattened = field.shell_harmonics(radii, mask=mask, flatten=True)
    unflattened = field.shell_harmonics(radii, mask=mask, flatten=False)

    assert flattened.irreps == o3.Irreps("0e")
    assert flattened.data.shape == (2, len(radii), flattened.irreps.dim)
    assert unflattened.data.shape == (3, 1, 1, len(radii), unflattened.irreps.dim)

    active_indices = torch.tensor([0, 2])
    for idx, active in enumerate(active_indices):
        expected = field.coeffs[0][active, :, :].sum(dim=0) * field._spherical_jn(0, radii[0] * field._k_values_for_l(0))
        assert torch.allclose(flattened.data[idx, 0], expected, atol=1e-6)
        assert torch.allclose(unflattened.data[active, 0, 0, 0], expected, atol=1e-6)