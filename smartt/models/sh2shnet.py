# sh2sh_e3nn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import Gate

# ---------- helpers ----------

def sh_irreps_lmax(lmax: int, mul: int = 1, include_l0: bool = True) -> o3.Irreps:
    """Irreps string like '1x0e + 1x1o + 1x2e + ...' up to lmax."""
    parts = []
    start = 0 if include_l0 else 1
    for l in range(start, lmax + 1):
        parity = "e" if (l % 2 == 0) else "o"  # Y_lm parity (-1)^l
        parts.append(f"{mul}x{l}{parity}")
    return o3.Irreps(" + ".join(parts) if parts else "0e")

def sh_dim_lmax(lmax: int, include_l0: bool = True) -> int:
    start = 0 if include_l0 else 1
    return sum(2 * l + 1 for l in range(start, lmax + 1))

def heat_kernel_alpha(lmax: int, sigma: float) -> torch.Tensor:
    """Diagonal 'blur' per-ℓ (same for all m): exp(-sigma^2 * ℓ(ℓ+1))."""
    alphas = []
    for l in range(lmax + 1):
        a = math.exp(-(sigma**2) * l * (l + 1))
        alphas += [a] * (2 * l + 1)
    return torch.tensor(alphas, dtype=torch.get_default_dtype())

# ---------- a gated equivariant block ----------

class GatedBlock(nn.Module):
    """
    Linear -> Gate(scalars & gates modulate non-scalars)
    Returns features with same 'content' irreps: (scalars 0e) + (non-scalars l>0).
    """
    def __init__(self, in_irreps: o3.Irreps, lmax_hidden: int, mul_non_scalar: int = 4, scalars_hidden: int = 16):
        super().__init__()

        # non-scalar hidden irreps (all l>=1 up to lmax_hidden)
        non_scalar_irreps = sh_irreps_lmax(lmax_hidden, mul=mul_non_scalar, include_l0=False)

        # scalars (0e) used both as outputs and as "gates" (one gate per non-scalar irrep)
        self.scalar_irreps = o3.Irreps(f"{scalars_hidden}x0e")
        self.n_gates = non_scalar_irreps.num_irreps
        self.gate_irreps = o3.Irreps(f"{self.n_gates}x0e")

        # produce (scalars + gates + non-scalars), then Gate applies activations & gating
        self.lin = o3.Linear(in_irreps, self.scalar_irreps + self.gate_irreps + non_scalar_irreps)

        # Gate: SiLU on scalars, Sigmoid on gates, gates modulate non-scalars
        # Note: one activation per irrep type, not per multiplicity
        self.gate = Gate(
            self.scalar_irreps,
            [torch.nn.SiLU()],  # single activation for the scalar irrep
            self.gate_irreps,
            [torch.nn.Sigmoid()],  # single activation for the gate irrep
            non_scalar_irreps
        )

        # Expose output irreps (scalars + non-scalars after gating)
        self.out_irreps = self.scalar_irreps + non_scalar_irreps

    def forward(self, x):
        x = self.lin(x)
        x = self.gate(x)
        return x

# ---------- full SH->SH model ----------

class SH2SHNet(nn.Module):
    """
    Rotation-equivariant network mapping SH coeffs (ℓ<=lmax_in) → SH coeffs (ℓ<=lmax_out).
    Uses a few gated equivariant blocks; allows internal bandwidth expansion to lmax_hidden.
    """
    def __init__(
        self,
        lmax_in: int = 8,
        lmax_out: int = 8,
        lmax_hidden: int = 12,      # allow bandwidth expansion inside the net (good with nonlinearities)
        mul_non_scalar: int = 4,    # width per ℓ>0
        scalars_hidden: int = 16,   # number of 0e scalars per block
        n_blocks: int = 3 
    ):
        super().__init__()
        self.in_irreps  = sh_irreps_lmax(lmax_in,  mul=1, include_l0=True)
        self.out_irreps = sh_irreps_lmax(lmax_out, mul=1, include_l0=True)

        blocks = []
        current = self.in_irreps
        for _ in range(n_blocks):
            blk = GatedBlock(
                in_irreps=current,
                lmax_hidden=lmax_hidden,
                mul_non_scalar=mul_non_scalar,
                scalars_hidden=scalars_hidden
            )
            blocks.append(blk)
            current = blk.out_irreps

        self.blocks = nn.ModuleList(blocks)
        self.final  = o3.Linear(current, self.out_irreps)

    def forward(self, x):
        # x: (B, dim_in) in the SAME irrep basis/order as self.in_irreps
        for blk in self.blocks:
            x = blk(x)
        return self.final(x)

# ---------- rotation helpers ----------

def rotate_coeff_batch(x: torch.Tensor, irreps: o3.Irreps, R: torch.Tensor) -> torch.Tensor:
    """
    Rotate SH coefficients using the representation D(R) for the given irreps.
    x: (B, dim)   R: (3, 3) rotation matrix
    """
    D = irreps.D_from_matrix(R)  # (dim, dim), consistent with e3nn's real basis
    return x @ D.T                # (B, dim)  (right-multiply by D^T)