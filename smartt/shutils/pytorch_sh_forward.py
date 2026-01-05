"""
pytorch_sh_forward
====================

This module provides a PyTorch implementation that mimics the behaviour of
``SphericalHarmonics.forward`` from the ``mumott`` package.  It evaluates
band‑limited spherical harmonic expansions on the sample directions provided
by a :class:`ProbedCoordinates` object and approximates the integral over
detector segments by a simple average of function values at the supplied
sampling points.  The implementation uses real, 4π‑normalised spherical
harmonics so that its outputs match those produced by the corresponding
Fortran/NumPy implementation in ``mumott`` when the integral is evaluated
by summation.  All computations are vectorised using PyTorch operations,
making this routine fully compatible with autograd and GPU acceleration.

Classes
-------

``TorchSphericalHarmonicsForward``
    Encapsulates a batched forward operator for spherical harmonic
    coefficients.  Given a batch of coefficient vectors and a
    ``ProbedCoordinates`` instance, it returns the projections of the
    corresponding spherical functions onto each detector segment by
    averaging values at the sampling points along the segment.

Example
-------

.. code-block:: python

    import numpy as np
    import torch
    from mumott import ProbedCoordinates
    from pytorch_sh_forward import TorchSphericalHarmonicsForward
    from evaluate_sh import _generate_lm_list

    # Create random probed coordinates (N projections, M detector segments, I samples per segment)
    N, M, I = 2, 3, 4
    # Sample directions uniformly on the sphere for demonstration
    rng = np.random.default_rng(0)
    # Sample cos(theta) uniformly in [-1, 1] and phi uniformly in [0, 2π)
    u = rng.uniform(-1.0, 1.0, size=(N, M, I))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(N, M, I))
    theta = np.arccos(u)
    # Convert spherical to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    coords = ProbedCoordinates()
    coords.vector = np.stack([x, y, z], axis=-1)  # shape (N, M, I, 3)

    # ell_max = 8 (using only even ℓ), so there are 45 coefficients
    ell_max = 8
    num_coeffs = (ell_max // 2 + 1) ** 2  # 1+5+9+13+17 = 45
    # Generate random coefficients for each projection
    coeffs = torch.randn(N, num_coeffs, dtype=torch.float64, requires_grad=True)

    # Construct the forward operator
    sh_forward = TorchSphericalHarmonicsForward(coords, ell_max=ell_max)
    # Evaluate the forward projection
    out = sh_forward(coeffs)
    # ``out`` has shape (N, M) containing the approximate integrals over each detector segment
    # Backpropagate through the output
    loss = out.pow(2).mean()
    loss.backward()
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch

from smartt.shutils.evaulate_sh import _generate_lm_list, associated_legendre


class TorchSphericalHarmonicsForward:
    """Batched forward projection operator for spherical harmonics using PyTorch.

    This class evaluates band‑limited spherical harmonic expansions on
    sampling points given by a :class:`ProbedCoordinates` instance and
    approximates the integral over each detector segment by a simple
    average of the function values at those points.  The
    implementation uses **real, 4π‑normalised spherical harmonics** so
    that its outputs match those produced by the Fortran/NumPy
    implementation in ``mumott`` when the integral is approximated by
    summation.  All computations are vectorised using PyTorch and are
    fully compatible with autograd and GPU acceleration.

    Parameters
    ----------
    coords : ProbedCoordinates
        A container providing the sampling points on the sphere.  The
        attribute ``coords.vector`` is expected to have shape
        ``(N, M, I, 3)``, where ``N`` is the number of tomographic
        projections, ``M`` is the number of detector segments per
        projection, ``I`` is the number of sample points per segment,
        and the final dimension contains the Cartesian coordinates of
        each point.  The ``ProbedCoordinates`` class from ``mumott``
        satisfies this contract.
    ell_max : int, optional
        The maximum spherical harmonic degree.  Only even degrees up to
        ``ell_max`` (inclusive) are used, resulting in a coefficient
        vector of length ``(ell_max//2 + 1)**2``.  Default is 0.

    Notes
    -----
    The real harmonics constructed internally follow

    .. math::

       Y_{\ell m}^{(4\pi)}(\theta,\phi)
         = \sqrt{(2-\delta_{m0})(2\ell+1)\,\frac{(\ell-m)!}{(\ell+m)!}}\,
           P_{\ell}^m(\cos\theta)
         \begin{cases}
           \cos(m\phi), & m > 0,\\
           \sin(|m|\phi), & m < 0,\\
           1, & m = 0,
         \end{cases}

    where ``P_{\ell}^m`` is the associated Legendre function.  These
    harmonics satisfy the orthogonality relation

    .. math::

       \int_{S^2} Y_{\ell m}^{(4\pi)}\,(Y_{\ell' m'}^{(4\pi)})\,d\Omega
       = 4\pi\,\delta_{\ell\ell'}\delta_{mm'}.

    The forward projection sums the product of each harmonic and its
    coefficient over all degrees and orders, averages over the ``I``
    sample points in each detector segment, and returns a tensor of
    shape ``(N, M)``.
    """

    def __init__(self, coords: "ProbedCoordinates", *, ell_max: int = 0) -> None:
        # Store coordinates and bandlimit
        self.coords = coords
        self.ell_max = ell_max
        # Generate the list of (ℓ, m) pairs for even ℓ up to ell_max
        self.lm_list: List[Tuple[int, int]] = _generate_lm_list(ell_max)
        self.num_coeffs: int = len(self.lm_list)
        # Precompute the normalisation factors for each (ℓ, |m|)
        # These factors are constants (not part of the autograd graph)
        self._norm_factors = {}
        for l, m in self.lm_list:
            m_abs = abs(m)
            two_minus_delta = 1 if m_abs == 0 else 2  # 2 - δ_{m0}
            num = math.factorial(l - m_abs)
            den = math.factorial(l + m_abs)
            factor = math.sqrt(two_minus_delta * (2 * l + 1) * (num / float(den)))
            self._norm_factors[(l, m_abs)] = factor

    def _compute_basis(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Compute the spherical harmonic basis on the sampling points.

        **Deprecated:** This method is retained for backward compatibility
        but is no longer used internally.  It constructs the full basis
        tensor of shape ``(N, M, I, num_coeffs)`` and thus incurs a
        large memory overhead.  The new implementation in
        :meth:`__call__` computes the projection in a streaming fashion
        without materialising this basis tensor.  Use
        :meth:`~TorchSphericalHarmonicsForward.__call__` directly for
        efficient evaluation.
        """
        _, theta_np, phi_np = self.coords.to_spherical
        theta = torch.as_tensor(theta_np, dtype=dtype, device=device)
        phi = torch.as_tensor(phi_np, dtype=dtype, device=device)
        x = torch.cos(theta)
        shape = theta.shape
        basis = torch.zeros((*shape, self.num_coeffs), dtype=dtype, device=device)
        for idx, (l, m) in enumerate(self.lm_list):
            m_abs = abs(m)
            P = associated_legendre(l, m_abs, x)
            factor = self._norm_factors[(l, m_abs)]
            if m > 0:
                Y = factor * P * torch.cos(m_abs * phi)
            elif m < 0:
                Y = factor * P * torch.sin(m_abs * phi)
            else:
                Y = factor * P
            basis[..., idx] = Y
        return basis

    def __call__(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply the forward projection to a batch of coefficient vectors.

        Parameters
        ----------
        coeffs : torch.Tensor
            A tensor of shape ``(N, C)`` where ``C`` is the number of
            coefficients (45 for ``ell_max=8``) and ``N`` matches the
            first dimension of ``coords.vector`` (the number of
            projections).  Each row contains the real spherical
            harmonic coefficients for one tomographic projection.  The
            coefficients should be ordered as in
            :func:`evaluate_sh._generate_lm_list`, i.e., grouped by
            ascending even ``ℓ`` and within each degree by ascending
            order ``m`` from ``-ℓ`` to ``ℓ``.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(N, M)`` giving the approximate
            integral of each spherical function over each detector
            segment.  Autograd gradients propagate back to
            ``coeffs``.

        Raises
        ------
        ValueError
            If the coefficient tensor has an incompatible shape.
        """
        if coeffs.dim() != 2:
            raise ValueError("coeffs must have shape (N, C)")
        N, C = coeffs.shape
        if C != self.num_coeffs:
            raise ValueError(
                f"coeffs has {C} coefficients per function, expected {self.num_coeffs} "
                f"for ell_max={self.ell_max}"
            )
        # Ensure that the number of coefficient rows matches the number of projections
        if N != self.coords.shape[0]:
            raise ValueError(
                f"coeffs has {N} rows but coords has {self.coords.shape[0]} projections"
            )
        # Extract spherical coordinates from probed coordinates.
        # ``to_spherical`` returns (r, theta, phi) as NumPy arrays of
        # shape (N, M, I).  We convert to torch for autograd.
        theta_np, phi_np = self.coords[:, 0], self.coords[:, 1]
        theta = torch.as_tensor(theta_np, dtype=coeffs.dtype, device=coeffs.device)
        phi = torch.as_tensor(phi_np, dtype=coeffs.dtype, device=coeffs.device)
        # Compute cos(theta) once for all sample points.  This yields a
        # tensor of shape (N, M, I).
        x = torch.cos(theta)
        # Initialise the accumulator for the function values.  Values
        # will have shape (N, M, I).  We allocate a new tensor rather
        # than building a (N, M, I, num_coeffs) basis to save memory.
        values = torch.zeros_like(x)
        # Loop over each (ℓ, m) pair.  For each harmonic, compute
        # the associated Legendre polynomial P_l^{|m|}(x), multiply by
        # the appropriate trigonometric function of phi, scale by the
        # normalisation factor, and accumulate the coefficient‑weighted
        # contribution into ``values``.
        for idx, (l, m) in enumerate(self.lm_list):
            m_abs = abs(m)
            # Compute P_l^{|m|}(x) for all sample points; shape (N, M, I).
            P = associated_legendre(l, m_abs, x)
            # Retrieve the 4π normalisation factor
            factor = self._norm_factors[(l, m_abs)]
            # Build the real spherical harmonic at each sample point.
            if m > 0:
                Y = factor * P * torch.cos(m_abs * phi)
            elif m < 0:
                Y = factor * P * torch.sin(m_abs * phi)
            else:
                Y = factor * P
            # Multiply by the corresponding coefficient for each
            # projection ``n``.  ``coeffs[:, idx]`` has shape
            # (N,) and we reshape it to (N, 1, 1) for broadcasting
            # across M and I dimensions.
            coeff_n = coeffs[:, idx].reshape(N, 1, 1)
            values = values + Y * coeff_n
        # Finally, approximate the integral by averaging along the
        # sampling dimension I
        output = values.mean(dim=-1)
        return output


def _test_equivalence(num_tests: int = 3) -> None:
    """Internal test comparing the PyTorch implementation with a NumPy reference.

    This test generates random coefficients and random sampling grids and
    checks that the PyTorch forward projection produces the same results
    as an explicit evaluation using the orthonormal spherical harmonics
    multiplied by :math:`\sqrt{4\pi}`.  The reference uses NumPy to
    evaluate the harmonics, but does not require SciPy.  If SciPy is
    available, one could alternatively verify against ``sph_harm``.
    The test is not intended to be exhaustive; it merely demonstrates
    consistency for a handful of cases.

    Parameters
    ----------
    num_tests : int, optional
        The number of random test cases to generate.  Default is 3.

    Raises
    ------
    AssertionError
        If any test case produces a mismatch beyond numerical
        tolerance.
    """
    import numpy as np

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    for _ in range(num_tests):
        # Randomly choose the geometry dimensions
        N = rng.integers(1, 3) + 1  # number of projections (1 or 2)
        M = rng.integers(1, 4) + 1  # number of detector segments (1–4)
        I = rng.integers(1, 4) + 1  # number of samples per segment (1–4)
        # Generate random spherical coordinates for each sample point
        u = rng.uniform(-1.0, 1.0, size=(N, M, I))
        phi = rng.uniform(0.0, 2.0 * np.pi, size=(N, M, I))
        theta = np.arccos(u)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        # Instantiate ProbedCoordinates with the random vectors
        from mumott import ProbedCoordinates as _ProbedCoordinates  # type: ignore
        coords = _ProbedCoordinates()
        coords.vector = np.stack([x, y, z], axis=-1)
        # ell_max = 8 → 45 coefficients
        ell_max = 8
        num_coeffs = (ell_max // 2 + 1) ** 2
        # Random coefficients for each projection
        coeffs = torch.randn(N, num_coeffs, dtype=torch.float64, requires_grad=True)
        # Build the PyTorch forward operator
        sh_forward = TorchSphericalHarmonicsForward(coords, ell_max=ell_max)
        # Evaluate with PyTorch
        out_torch = sh_forward(coeffs)
        # Evaluate with NumPy reference: compute orthonormal basis and scale by sqrt(4π)
        # Compute basis per sample point and per (l,m)
        lm_list = _generate_lm_list(ell_max)
        # Flatten sampling points for convenience
        theta_flat = theta.reshape(-1)
        phi_flat = phi.reshape(-1)
        x_flat = np.cos(theta_flat)
        # Compute orthonormal real spherical harmonics using evaluate_sh formulas in NumPy
        # Then multiply by sqrt(4π) to obtain 4π normalised values
        # Preallocate basis array: shape (num_points, num_coeffs)
        num_points = theta_flat.size
        basis_np = np.zeros((num_points, num_coeffs), dtype=np.float64)
        for idx, (l, m) in enumerate(lm_list):
            m_abs = abs(m)
            # Compute associated Legendre P_l^{|m|}(cos θ)
            # Using recurrence from evaluate_sh.associated_legendre via NumPy
            # Convert x_flat to torch tensor for reuse of associated_legendre
            x_t = torch.as_tensor(x_flat, dtype=torch.float64)
            P = associated_legendre(l, m_abs, x_t).numpy()
            # Compute orthonormal normalisation factor K
            num = math.factorial(l - m_abs)
            den = math.factorial(l + m_abs)
            K = math.sqrt((2 * l + 1) / (4.0 * math.pi) * (num / float(den)))
            if m > 0:
                Y = math.sqrt(2.0) * K * P * np.cos(m_abs * phi_flat)
            elif m < 0:
                Y = math.sqrt(2.0) * K * P * np.sin(m_abs * phi_flat)
            else:
                Y = K * P
            basis_np[:, idx] = Y
        # Convert to 4π normalised by multiplying by sqrt(4π)
        basis_np *= math.sqrt(4.0 * math.pi)
        basis_np = basis_np.reshape(N, M, I, num_coeffs)
        # Compute expected output via NumPy: dot with coefficients and mean over samples
        coeffs_np = coeffs.detach().numpy()
        out_ref = np.zeros((N, M), dtype=np.float64)
        for n in range(N):
            for m_i in range(M):
                # Dot product over coefficients for each sample point
                vals = np.dot(basis_np[n, m_i, :, :], coeffs_np[n])
                out_ref[n, m_i] = vals.mean()
        # Compare PyTorch and NumPy results
        diff = torch.max(torch.abs(out_torch - torch.as_tensor(out_ref)))
        tol = 1e-8
        assert diff.item() < tol, (
            f"Forward projection mismatch (diff={diff.item()}):\n"
            f"out_torch={out_torch}\n"
            f"out_ref={out_ref}"
        )


if __name__ == "__main__":
    # Run internal tests when executed as a script
    _test_equivalence()
    print("All tests passed.")