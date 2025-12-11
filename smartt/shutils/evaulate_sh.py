from __future__ import annotations

import math
from typing import Iterable, List, Tuple, Union, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def _generate_lm_list(max_l: int) -> List[Tuple[int, int]]:
    """Generate a list of (ℓ, m) pairs for even ℓ up to ``max_l``.

    The pairs are ordered first by ascending ℓ and within each ℓ by ascending
    m from ``-ℓ`` to ``ℓ``.  Only even ℓ values are included.  This ordering
    matches the assumed layout of the input coefficient vector of length 45.

    Args:
        max_l: The maximum degree ℓ (inclusive).

    Returns:
        A list of (ℓ, m) tuples.
    """
    lm_list: List[Tuple[int, int]] = []
    for ell in range(0, max_l + 1, 2):  # include only even bands
        for m in range(-ell, ell + 1):
            lm_list.append((ell, m))
    return lm_list


def normalization_factor(l: int, m: int) -> float:
    r"""Compute the normalization factor for real spherical harmonics.

    The normalization factor ``N_{ℓm}`` for the complex spherical harmonic
    ``Y_ℓ^m`` is given by

    .. math::

       N_{ℓm} = \sqrt{\frac{2ℓ + 1}{4π}\,\frac{(ℓ - m)!}{(ℓ + m)!}}.

    This factor is the same for the real basis except that an additional
    ``√2`` is applied externally for ``m ≠ 0`` when forming the real
    harmonics.  The returned value is therefore the factor that multiplies
    ``P_ℓ^m(cosθ)`` in the definition of real spherical harmonics.

    Args:
        l: Degree ℓ (non-negative integer).
        m: Order m (non-negative integer).  The absolute value should be
            provided; the normalization is even in m.

    Returns:
        A floating point normalization constant.
    """
    # Use Python's math.factorial for integer factorials.  This is not part of
    # the autograd graph because it depends only on integer inputs.
    numerator = math.factorial(l - m)
    denominator = math.factorial(l + m)
    factor = (2 * l + 1) / (4.0 * math.pi) * (numerator / float(denominator))
    return math.sqrt(factor)


def associated_legendre(l: int, m: int, x: torch.Tensor) -> torch.Tensor:
    """Evaluate the associated Legendre polynomial ``P_ℓ^m(x)``.

    This implementation uses a stable recurrence based on ``P_m^m`` and
    ``P_{m+1}^m``.  The recurrence follows from the standard relations:

    .. math::

       P_m^m(x) &= (-1)^m (2m - 1)!! (1 - x^2)^{m/2},\\
       P_{m+1}^m(x) &= x (2m + 1) P_m^m(x),\\
       P_n^m(x) &= \frac{(2n - 1) x P_{n-1}^m(x) - (n + m - 1) P_{n-2}^m(x)}{n - m}

    for ``n > m + 1``.

    Args:
        l: Degree ℓ (integer ≥ m).
        m: Order m (integer ≥ 0).
        x: A tensor of shape ``(*,)`` containing the argument ``x = cosθ``.  The
           returned tensor will have the same shape.

    Returns:
        A tensor of the same shape as ``x`` containing ``P_ℓ^m(x)``.
    """
    if m < 0 or l < m:
        raise ValueError("Require 0 ≤ m ≤ ℓ in associated_legendre")

    # Base case P_m^m(x)
    # P_0^0(x) = 1
    # P_1^1(x) = -(2*1 - 1) sqrt(1 - x^2) * P_0^0(x) = -1 * sqrt(1 - x^2)
    P_mm = torch.ones_like(x)
    if m > 0:
        P_mm = torch.ones_like(x)
        # Compute (1 - x**2)**0.5 once
        # We clip small negative values that may arise from numerical rounding
        # errors to avoid NaNs during the sqrt.  This is still differentiable
        # since x is differentiable.  It's generally safe because (1 - x**2) is
        # non-negative for |x| ≤ 1.
        # to ensure numerical stability we use torch.relu.
        one_minus_x2 = torch.relu(1.0 - x * x)
        root = torch.sqrt(one_minus_x2)
        for k in range(1, m + 1):
            P_mm = - (2 * k - 1) * root * P_mm
    if l == m:
        return P_mm
    # P_{m+1}^m(x) = x (2m + 1) P_m^m(x)
    P_m1m = (2 * m + 1) * x * P_mm
    if l == m + 1:
        return P_m1m
    # Now use recurrence to compute P_n^m for n = m+2,...,l
    P_n2 = P_mm  # P_{m}^{m}
    P_n1 = P_m1m  # P_{m+1}^{m}
    # We iteratively compute P_{n}^m using P_{n-1}^m and P_{n-2}^m.
    for n in range(m + 2, l + 1):
        # (2n - 1) * x * P_{n-1}^m - (n + m - 1) * P_{n-2}^m
        term1 = (2 * n - 1) * x * P_n1
        term2 = (n + m - 1) * P_n2
        P_n = (term1 - term2) / (n - m)
        P_n2, P_n1 = P_n1, P_n
    return P_n1


"""
pytorch_sh_quadrature
=====================

This module provides a batched PyTorch implementation of the
spherical‑harmonic forward operator that approximates the
integration of a band‑limited spherical function along detector
segments using classical quadrature rules.  It is designed to
mimic the behaviour of the ``SphericalHarmonics.forward`` method
in the ``mumott`` package while remaining GPU friendly and
memory efficient.

The function :func:`forward_quadrature` accepts batches of
spherical harmonic coefficients and corresponding sampling points
for each detector segment.  For each coefficient set, it
evaluates the spherical harmonic expansion at the sample points,
applies one of three quadrature schemes (Simpson, trapezoidal or
midpoint), and returns the approximate integrals of the function
along each segment.  The implementation exploits PyTorch’s
vectorisation and broadcasting to perform the computation across
arbitrary spatial dimensions in a single pass, minimising
intermediate memory usage.

Example
-------

.. code-block:: python

    import numpy as np
    import torch
    from mumott import ProbedCoordinates
    from pytorch_sh_quadrature import forward_quadrature

    # Create a simple geometry: N projections, M detector segments, I samples per segment
    N, M, I = 2, 3, 5
    # Create random unit vectors on the sphere for demonstration
    rng = np.random.default_rng(0)
    theta = np.arccos(rng.uniform(-1.0, 1.0, size=(N, M, I)))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(N, M, I))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    coords = ProbedCoordinates()
    coords.vector = np.stack([x, y, z], axis=-1)

    # Use ell_max = 8 (even degrees only ⇒ 45 coefficients)
    ell_max = 8
    # Number of coefficients: (ell_max//2 + 1) * (ell_max + 1)
    C = (ell_max // 2 + 1) * (ell_max + 1)
    # Generate random coefficients with additional spatial dimensions H×W
    H, W = 4, 5
    coeffs = torch.randn(N, H, W, C, dtype=torch.float64, requires_grad=True)

    # Compute the forward projection using Simpson quadrature
    out = forward_quadrature(coords, coeffs, ell_max=ell_max, mode='simpson')
    # ``out`` has shape (N, H, W, M) containing the approximate integrals
    # of the spherical functions along each segment for each set of
    # coefficients.
    loss = out.pow(2).sum()
    loss.backward()

    # ``coeffs.grad`` now contains gradients of the loss w.r.t. the input coefficients

"""




def _compute_quadrature_weights(
    num_samples: int,
    dtype: torch.dtype,
    device: torch.device,
    mode: str = "simpson",
) -> torch.Tensor:
    """Return a 1‑D tensor of quadrature weights for ``num_samples`` points.

    Parameters
    ----------
    num_samples : int
        The number of sample points along each detector segment.  When
        Simpson’s rule is selected, this must be an odd integer; if it
        is even or less than three, the function falls back to the
        trapezoidal rule.
    dtype : torch.dtype
        The desired dtype of the returned tensor.
    device : torch.device
        The device on which to allocate the weights.
    mode : {'simpson', 'trapezoid', 'midpoint'}, optional
        The quadrature scheme to use.  ``'simpson'`` employs Simpson’s
        rule, which gives third‑order accuracy but requires an odd
        number of samples.  ``'trapezoid'`` uses the trapezoidal rule
        (second‑order accurate).  ``'midpoint'`` uses a simple
        rectangle rule with equal weights at all sample points.

    Returns
    -------
    torch.Tensor
        A 1‑D tensor of length ``num_samples`` containing the weights.
        The weights are normalised such that their sum equals 1, so the
        approximate integral is simply the dot product of the weights
        with the sampled function values.
    """
    # Simpson's rule: requires odd number of samples and at least 3 points
    if mode.lower() == "simpson" and num_samples % 2 == 1 and num_samples >= 3:
        h = 1.0 / (num_samples - 1)
        # Simpson weights: 1,4,2,4,...,4,1
        weights = [1.0 if i == 0 or i == num_samples - 1 else (4.0 if i % 2 == 1 else 2.0)
                   for i in range(num_samples)]
        scale = h / 3.0
        weights = [w * scale for w in weights]
        total = sum(weights)
        weights = [w / total for w in weights]
    # Trapezoidal rule: requires at least 2 points
    elif mode.lower() == "trapezoid" and num_samples >= 2:
        h = 1.0 / (num_samples - 1)
        weights = [1.0] + [2.0] * (num_samples - 2) + [1.0]
        weights = [w * (h / 2.0) for w in weights]
        total = sum(weights)
        weights = [w / total for w in weights]
    else:
        # Midpoint or fallback: equal weights
        weights = [1.0 / num_samples] * num_samples
    return torch.tensor(weights, dtype=dtype, device=device)


def forward_quadrature(
    coords: "ProbedCoordinates",
    coeffs: torch.Tensor,
    *,
    ell_max: int,
    mode: str = "simpson",
    projection_matrix: Union[NDArray, torch.Tensor, None] = None,
) -> torch.Tensor:
    """Evaluate spherical harmonic expansions and integrate along detector segments.

    Parameters
    ----------
    coords : ProbedCoordinates
        A container whose ``vector`` attribute has shape ``(N, M, I, 3)``
        giving the Cartesian coordinates of the sample points along
        each detector segment.  ``N`` is the number of tomographic
        projections, ``M`` the number of detector segments per
        projection, and ``I`` the number of sampling points per
        segment.  This parameter is ignored if ``projection_matrix`` is
        provided.
    coeffs : torch.Tensor
        A tensor of spherical harmonic coefficients of shape
        ``(N, H, W, C)``, where ``H`` and ``W`` are arbitrary spatial
        dimensions (for example, detector row and column).  The last
        dimension ``C`` must equal ``(ell_max//2 + 1) * (ell_max + 1)`` and
        store coefficients ordered as returned by
        :func:`_generate_lm_list`, i.e. with degrees ``ℓ``
        ranging over even values ``0, 2, …, ell_max`` and orders ``m``
        ranging from ``-ℓ`` to ``ℓ`` for each ``ℓ``.  Autograd gradients
        will propagate back to ``coeffs``.
    ell_max : int
        The maximum spherical harmonic degree.  Only even degrees up to
        this value (inclusive) are used.  This parameter is ignored if
        ``projection_matrix`` is provided.
    mode : {'simpson', 'trapezoid', 'midpoint'}, optional
        The quadrature scheme to approximate the integral along each
        segment.  If ``mode=='simpson'`` and the number of samples is
        even or less than three, the trapezoidal rule is used instead.
        This parameter is ignored if ``projection_matrix`` is provided.
    projection_matrix : np.ndarray or torch.Tensor, optional
        A precomputed projection matrix of shape ``(N, M, C)`` containing
        the integrated spherical harmonic basis values for each projection
        and detector segment.  If provided, this matrix is used directly
        instead of computing the integration from ``coords``.  This is
        useful for exact matching with mumott's ``SphericalHarmonics``
        class, which uses adaptive quadrature for more accurate integration.
        The matrix can be obtained from ``basis_set.projection_matrix``
        where ``basis_set`` is a mumott ``SphericalHarmonics`` instance.

    Returns
    -------
    torch.Tensor
        A tensor of shape ``(N, H, W, M)`` containing the approximate
        integrals of the spherical functions along each detector
        segment, where ``M`` matches the number of detector segments
        in ``coords.vector`` (the second dimension).  The output is
        differentiable with respect to ``coeffs``.

    Notes
    -----
    When ``projection_matrix`` is not provided, this function computes
    the weighted sum of 4π‑normalised real spherical harmonics evaluated
    at each sample point.  For each harmonic of degree ``ℓ`` and order
    ``m``, a precomputed normalisation factor is applied, followed by the
    associated Legendre polynomial ``P_ℓ^{|m|}`` evaluated at ``cos(θ)``,
    and a sine or cosine term of ``mφ``.  The weights defined by ``mode``
    are then applied along the sampling dimension before summing over
    ``ℓ,m``.
    
    For exact matching with mumott's output, pass the ``projection_matrix``
    from a mumott ``SphericalHarmonics`` instance.  This avoids numerical
    differences from different quadrature schemes, especially for
    high-frequency spherical harmonics (large ``ℓ``).
    """
    # Check coefficient tensor shape
    if coeffs.dim() < 3:
        raise ValueError(
            f"coeffs must have at least three dimensions (N, H, W, C), got {coeffs.shape}"
        )
    N = coeffs.shape[0]
    C = coeffs.shape[-1]
    
    # If projection_matrix is provided, use it directly
    if projection_matrix is not None:
        if isinstance(projection_matrix, torch.Tensor):
            pm = projection_matrix.to(dtype=coeffs.dtype, device=coeffs.device)
        else:
            # Assume numpy array
            pm = torch.as_tensor(projection_matrix, dtype=coeffs.dtype, device=coeffs.device)
        
        # Validate shape: projection_matrix should be (N, M, C)
        if pm.dim() != 3:
            raise ValueError(
                f"projection_matrix must have 3 dimensions (N, M, C), got {pm.dim()}"
            )
        if pm.shape[0] != N:
            raise ValueError(
                f"projection_matrix first dim ({pm.shape[0]}) doesn't match coeffs ({N})"
            )
        if pm.shape[2] != C:
            raise ValueError(
                f"projection_matrix last dim ({pm.shape[2]}) doesn't match coeffs ({C})"
            )
        
        # Compute forward: out[n,h,w,m] = sum_c coeffs[n,h,w,c] * pm[n,m,c]
        return torch.einsum("nhwc,nmc->nhwm", coeffs, pm)
    
    # Otherwise, compute projection matrix from coords
    # For even ℓ up to ell_max, the number of coefficients is (ell_max//2 + 1)*(ell_max + 1)
    expected_C = (ell_max // 2 + 1) * (ell_max + 1)
    if C != expected_C:
        raise ValueError(
            f"coeffs last dimension is {C}, expected {expected_C} for ell_max={ell_max}"
        )
    # Ensure coords.vector shape matches expected
    if not hasattr(coords, "vector"):
        raise ValueError("coords must have a 'vector' attribute")
    vec = coords.vector
    if vec.ndim != 4 or vec.shape[-1] != 3:
        raise ValueError(
            f"coords.vector must have shape (N, M, I, 3), got {vec.shape}"
        )
    if vec.shape[0] != N:
        raise ValueError(
            f"Mismatch between coords.vector first dimension {vec.shape[0]} and coeffs {N}"
        )
    _, M, I, _ = vec.shape
    # Convert spherical coordinates to torch tensors on same device/dtype as coeffs
    _, theta_np, phi_np = coords.to_spherical
    theta = torch.as_tensor(theta_np, dtype=coeffs.dtype, device=coeffs.device)
    phi = torch.as_tensor(phi_np, dtype=coeffs.dtype, device=coeffs.device)
    # Compute cos(theta)
    x = torch.cos(theta)
    # Quadrature weights along the sampling dimension
    w = _compute_quadrature_weights(I, coeffs.dtype, coeffs.device, mode)
    # Generate list of (ℓ,m) pairs for even ℓ and precompute normalisation factors
    lm_list = _generate_lm_list(ell_max)
    num_coeffs = len(lm_list)
    # For even ℓ up to ell_max, the number of coefficients should be
    # (ell_max//2 + 1) * (ell_max + 1).  Verify this and warn if mismatch.
    expected_total = (ell_max // 2 + 1) * (ell_max + 1)
    if C != expected_total:
        raise ValueError(
            f"coeffs last dimension is {C}, expected {expected_total} for ell_max={ell_max}."
        )
    if num_coeffs != C:
        # This should not happen unless generate_even_lm_list is inconsistent.
        raise RuntimeError(
            f"Generated {num_coeffs} (ℓ,m) pairs but coeffs has {C} entries."
        )
    # Precompute normalisation factors for each (ℓ, |m|)
    norm_factors = {}
    for l, m in lm_list:
        m_abs = abs(m)
        two_minus_delta = 1 if m_abs == 0 else 2
        num = math.factorial(l - m_abs)
        den = math.factorial(l + m_abs)
        norm = math.sqrt(two_minus_delta * (2 * l + 1) * (num / float(den)))
        norm_factors[(l, m_abs)] = norm
    # Preallocate integrated basis array Y_int with shape (N, M, C)
    Y_int = torch.zeros(N, M, C, dtype=coeffs.dtype, device=coeffs.device)
    # Compute weighted integrals for each (ℓ,m)
    for idx, (l, m) in enumerate(lm_list):
        m_abs = abs(m)
        # Associated Legendre polynomial P_l^|m|(x)
        P = associated_legendre(l, m_abs, x)
        factor = norm_factors[(l, m_abs)]
        # Cancel the Condon-Shortley phase (-1)^|m| that is built into
        # our associated_legendre function to match mumott's convention
        cs_correction = (-1) ** m_abs
        if m > 0:
            Y = cs_correction * factor * P * torch.cos(m_abs * phi)
        elif m < 0:
            Y = cs_correction * factor * P * torch.sin(m_abs * phi)
        else:
            Y = factor * P
        # Weighted integration along I
        # w shape (I,) → unsqueeze to (1, 1, I) for broadcast
        Y_int[..., idx] = (Y * w.view(1, 1, -1)).sum(dim=2)
    # Multiply coefficients by integrated basis and sum over coefficient index C
    # coeffs: (N, H, W, C), Y_int: (N, M, C)
    # Output: (N, H, W, M)
    out = torch.einsum("nhwc,nmc->nhwm", coeffs, Y_int)
    return out