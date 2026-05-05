from .body import Embed, Encoder
from .heads import OutputHead, FTHead, RealPositiveHead, FTHeadReal
from flax import linen as nn
from typing import Callable


class ViT(nn.Module):
    num_layers: int
    d_model: int
    heads: int
    plattice_shape: tuple
    """
        Shape of patched lattice (Lx,Ly,...) 
    """
    extract_patches: Callable
    """
        Function for patchifying input
    """
    head: nn.module
    """
        Output head after encoder
    """
    kernel_shape: tuple = None  # shape for masked attention, if None, no masking
    expansion_factor: int = 4
    """
        Factor to expand model dimension in feedforward block 
    """
    transl_invariant: bool = True

    def setup(self):
        self.patches_and_embed = Embed(
            self.d_model, extract_patches=self.extract_patches
        )

        self.encoder = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            h=self.heads,
            plattice_shape=self.plattice_shape,
            expansion_factor=self.expansion_factor,
            transl_invariant=self.transl_invariant,
        )

        self.output = self.head

    def __call__(self, x):
        x = self.patches_and_embed(x)

        x = self.encoder(x)

        z = self.output(x)

        return z


def ViTndim(
    num_layers: int,
    d_model: int,
    heads: int,
    plattice_shape: tuple,
    extract_patches: Callable,
    kernel_shape: tuple = None,
    expansion_factor: int = 4,
    output_depth: int = 1,
    transl_invariant: bool = True,
):
    return ViT(
        num_layers=num_layers,
        d_model=d_model,
        heads=heads,
        plattice_shape=plattice_shape,
        extract_patches=extract_patches,
        head=OutputHead(d_model, output_depth=output_depth),
        kernel_shape=kernel_shape,
        expansion_factor=expansion_factor,
        transl_invariant=transl_invariant,
    )


def ViT_FT(
    num_layers: int,
    d_model: int,
    heads: int,
    plattice_shape: tuple,
    extract_patches: Callable,
    expansion_factor: int,
    q: tuple,
    compute_positions: Callable,
    kernel_shape: tuple = None,  # masked attention
    transl_invariant: bool = True,
):
    head = FTHead(d_model=d_model, q=q, compute_positions=compute_positions)

    return ViT(
        num_layers=num_layers,
        d_model=d_model,
        heads=heads,
        plattice_shape=plattice_shape,
        extract_patches=extract_patches,
        head=head,
        kernel_shape=kernel_shape,
        expansion_factor=expansion_factor,
        transl_invariant=transl_invariant,
    )


def ViT_FTReal(
    num_layers: int,
    d_model: int,
    heads: int,
    plattice_shape: tuple,
    extract_patches: Callable,
    expansion_factor: int,
    q: tuple,
    compute_positions: Callable,
    kernel_shape: tuple = None,
    transl_invariant: bool = True,
):
    head = FTHeadReal(d_model=d_model, q=q, compute_positions=compute_positions)

    return ViT(
        num_layers=num_layers,
        d_model=d_model,
        heads=heads,
        plattice_shape=plattice_shape,
        extract_patches=extract_patches,
        head=head,
        kernel_shape=kernel_shape,
        expansion_factor=expansion_factor,
        transl_invariant=transl_invariant,
    )


def ViT_Positive(
    num_layers: int,
    d_model: int,
    heads: int,
    plattice_shape: tuple,
    extract_patches: Callable,
    kernel_shape: tuple = None,
    expansion_factor: int = 4,
    transl_invariant: bool = True,
):
    return ViT(
        num_layers=num_layers,
        d_model=d_model,
        heads=heads,
        plattice_shape=plattice_shape,
        extract_patches=extract_patches,
        head=RealPositiveHead(d_model),
        kernel_shape=kernel_shape,
        expansion_factor=expansion_factor,
        transl_invariant=transl_invariant,
    )
