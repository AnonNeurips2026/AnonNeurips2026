from typing import Union
import einops
from netket.utils.types import Array
from netket.graph import Lattice
from netket.utils import struct
import jax.numpy as jnp

@struct.dataclass
class Extract1Dto1D(struct.Pytree):
    """
        Extract patch_size patches from the (..., nsites) input x, corresponding to 
        a spin configuration on a 1D lattice.
        Patches are made up of multiples of the unit cell.
        For example, with a single-site unit cell, patches are patch_size contiguous sites,
        whereas for a two-site unit cell, patches are patch_size*2 contiguous sites.
        Args:
            patch_size: int - size of the patches, in units of the lattice vector
            lattice_size: int - extent of the lattice, in units of the lattice vector
            nb: int - number of sites in the unit cell 
        Returns:
            x reshaped to (..., npatches, patch_size*nb) where x[..., i, :] corresponds to sites in i-th patch 
            along 1D latttice.
    """
    patch_size: int = struct.field(pytree_node=False)
    lattice_size: int = struct.field(pytree_node=False)
    nb: int = struct.field(pytree_node = False)

    def __call__(self, x: Array) -> Array:
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))  # add batch dimension
        b1 = self.patch_size #number of unit cells in patch
        Lp1 = self.lattice_size // b1 #number of patches
        x = x.reshape(x.shape[:-1]+(Lp1, b1, self.nb))
        x = einops.rearrange(x, "... Lp1 b1 nb -> ... Lp1 (b1 nb)")  # (..., npatches, nsites_per_patch)
        return x

    def __hash__(self):
        return hash((self.patch_size,self.lattice_size, self.nb))


@struct.dataclass
class Extract2Dto1D(struct.Pytree):
    """
        Extract (patch_shape[0],patch_shape[1]) patches from the (..., nsites) input x, corresponding to 
        a spin configuration on a 2D lattice.
        For use with a network which uses a 1D arrangement of patches (e.g ViT).
        Patches are made up of multiples of the unit cell.
        Assumes the lattice has the (default netket) site indexing s = n2*nb * n1*nb*L2 + mu , with n2, n1 coordinates of the unit cell along the a1 and a2 lattice
        vectors and mu the index of the site within the unit cell
        Args:
            patch_shape: tuple[int, int] - size of the patches along each dimension, in units of the lattice vectors
            lattice_shape: tuple[int, int] - shape of the unpatched lattice, in units of the lattice vectors, used to determine the number of patches
            along each dimension
            nb: int - number of sites in the unit cell
        Returns:
            x reshaped to (..., npatches, patch_size) where x[..., s, :] corresponds to sites in the s = np2 + np1*Lp2 patch,
            where np1, np2 are the coordinates along the ap1 and ap2 lattice vectors of the 2D patched lattice. 
    """
    patch_shape: tuple[int, int] = struct.field(pytree_node=False)
    lattice_shape: tuple[int, int] = struct.field(pytree_node=False)
    nb: int = struct.field(pytree_node = False)

    def __call__(self, x: Array) -> Array:
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))  # add batch dimension
        b1,b2  = self.patch_shape #patch size along each dimension
        Lp1, Lp2 = self.lattice_shape[0] // b1, self.lattice_shape[1]//b2 #number of patches along each dimension
        batch_dims = x.shape[:-1]
        x = x.reshape(batch_dims + (Lp1, b1, Lp2, b2, self.nb)) 
        x = einops.rearrange(x, "... Lp1 b1 Lp2 b2 nb -> ... (Lp1 Lp2) (b1 b2 nb)") #(batch_dims, npatches, patch_size)
        return x

    def __hash__(self):
        return hash((self.patch_shape, self.lattice_shape,self.nb))

@struct.dataclass
class Extract2Dto2D(struct.Pytree):
    """
        Extract (patch_shape[0],patch_shape[1]) patches from the (..., nsites) input x, corresponding to 
        a spin configuration on a 2D lattice.
        For use with a network which uses a 2D arrangement of patches (e.g ConvNext).
        Patches are made up of multiples of the unit cell.
        Assumes the lattice has the (default netket) site indexing s = n2*nb * n1*nb*L2 + mu, with n2, n1 coordinates along the a1 and a2 lattice
        vectors on the unpatched lattice and mu the index of the site within the unit cell
        Args:
            patch_shape: tuple[int, int] - size of the patches along each dimension, in units of the unit cell
            lattice_shape: tuple[int, int] - shape of the unpatched lattice, used to determine the number of patches
            along each dimension
            nb: int - number of sites in the unit cell
        Returns:
            x reshaped to (..., np1,np2, patch_size) where x[...,np1,np2, :] corresponds to sites in the (np1,np2) patch,
            where np1, np2 are the coordinates along the ap1 and ap2 lattice vectors of the 2D patched lattice. 
    """
    patch_shape: tuple[int, int] = struct.field(pytree_node=False)
    lattice_shape: tuple[int, int] = struct.field(pytree_node=False)
    nb: int = struct.field(pytree_node = False)

    def __call__(self, x: Array) -> Array:
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))  # add batch dimension
        b1, b2 = self.patch_shape #patch size along each dimension
        Lp1, Lp2 = self.lattice_shape[0]//b1, self.lattice_shape[1]//b2 #number of patches along each dimension
        batch_dims = x.shape[:-1]
        x = x.reshape(batch_dims + (Lp1, b1, Lp2, b2, self.nb))
        x = einops.rearrange(x, "... Lp1 b1 Lp2 b2 nb -> ... Lp1 Lp2 (b1 b2 nb)") #(batch_dims, np1, np2, patch_size)
        return x

    def __hash__(self):
        return hash((self.patch_shape, self.lattice_shape, self.nb))

@struct.dataclass
class Extract3Dto1D(struct.Pytree):
    """
        Extract (patch_shape[0],patch_shape[1],patch_shape[2]) patches from the (..., nsites) input x, corresponding to 
        a spin configuration on a 3D lattice.
        For use with a network which uses a 1D arrangement of patches (e.g ViT).
        Patches are made up of multiples of the unit cell.
        Assumes the lattice has the (default netket) site indexing s = n3*nb + n2*nb*L3 * n1*nb*L2*L3 +mu, with n1,n2,n3 coordinates along the a1,a2,a3 lattice
        vectors on the unpatched lattice, and mu the index of the site within the unit cell
        Args:
            patch_shape: tuple[int, int, int] - size of the patches along each dimension, in units of the lattice vectors
            lattice_shape: tuple[int, int, int] - shape of the unpatched lattice, in units of the lattice vectors used to determine the number of patches
            along each dimension
        Returns:
            x reshaped to (..., npatches, patch_size) where x[..., s, :] corresponds to sites in the s = np2 + np1*Lp2 patch,
            where np1, np2 are the coordinates along the ap1 and ap2 lattice vectors of the 2D patched lattice. 
    """
    patch_shape: tuple[int, int, int] = struct.field(pytree_node=False)
    lattice_shape: tuple[int, int, int] = struct.field(pytree_node=False)
    nb: int = struct.field(pytree_node = False)

    def __call__(self, x: Array) -> Array:
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))  # add batch dimension
        b1, b2, b3 = self.patch_shape #patch size along each dimension
        Lp1, Lp2, Lp3 = self.lattice_shape[0]//b1, self.lattice_shape[1]//b2, self.lattice_shape[2]//b3 #number of patches along each dimension
        batch_dims = x.shape[:-1]
        x = x.reshape(batch_dims+(Lp1, b1, Lp2, b2, Lp3, b3, self.nb)) #Assuming indexing of s = n3 + n2*L3 + n1*L2*L3
        x = einops.rearrange(x, '... L1 b1 L2 b2 L3 b3 nb -> ... (L1 L2 L3) (b1 b2 b3 nb)') #Rearrange and contract intra and inter-patch dimensions
        return x

    def __hash__(self):
        return hash((self.patch_shape, self.lattice_shape))

ExtractPatchesFunT = Union[
    Extract1Dto1D,
    Extract2Dto1D,
    Extract2Dto2D,
    Extract3Dto1D,
]


@struct.dataclass
class PositionsTo1D(struct.Pytree):
    """
    Return the positions of the centre of patches in shape (npatches, 2).
    The order of patches is the same as that from extract_patches_to1d
    """

    graph: Lattice = struct.field(pytree_node=False)
    extract_patches: ExtractPatchesFunT

    def __call__(self) -> Array:
        # (ndim,npatches,patch_size)
        patched_positions = self.extract_patches(
            jnp.asarray(self.graph.positions.transpose())
        )
        # (ndim,npatches), gives centre of sites making up patch
        patch_positions = jnp.mean(patched_positions, axis=-1)
        # (npatches,ndim)
        return patch_positions.transpose()

    def __hash__(self):
        return hash((self.graph, self.extract_patches))


@struct.dataclass
class PositionsTo2D(struct.Pytree):
    """
    Return the positions of the centre of patches in shape (npx,npy, 2).
    The order of patches is the same as that from extract_patches_to2d
    """

    graph: Lattice = struct.field(pytree_node=False)
    extract_patches: ExtractPatchesFunT

    def __call__(self) -> Array:
        # (ndim,npx,npy,patch_size)
        patched_positions = self.extract_patches(
            jnp.asarray(self.graph.positions.transpose())
        )
        # (ndim,npx,npy), gives centre of sites making up patch
        patch_positions = jnp.mean(patched_positions, axis=-1)
        # (npx,npy,ndim)
        return patch_positions.transpose((1, 2, 0))

    def __hash__(self):
        return hash((self.graph, self.extract_patches))


class Patching:
    """
        Class to define patching functions from graph and other parameters.
    """

    def __init__(self, graph: Lattice, output_dim: int, patch_shape: tuple = None):
        """
            Defines the function self.extract_patches which performs the patching operation on the (..., nsites) input x,
            as well as the function self.compute_positions which returns the positions of the patches. 
            If graph has a single-site unit cell, the patches are defined as 2**dim hypercubes, unless patch_shape is specified.
            If the graph has a multi-site unit cell, the patches are defined as the the sites in the unit cell by default, unless patch_shape is specified,
            in which case the patch_shape specifies the shape of unit cells making up the patch.
            output_dim specifies whether extract_patches returns a (..., npatches, patch_size) or (..., np1, np2, patch_size) output, for use for example with 
            the ViT or Convolutional architectures respectively.

            Args:
                graph: Lattice - The lattice on which the variables are defined.
                output_dim: int - The number of output patch dimensions, 1 for ViT or 2 for Convolutional Network
                patch_shape: tuple - [Optional] - The shape of the patches to use for a single-site unit cell lattice. If 
                not specified, the patches are defined as 2**dim hypercubes.
        """
        self.output_dim = output_dim  # Dimensions of patched lattice
        self.lattice_dim = graph.ndim
        self.n_sublattices = len(graph.site_offsets)  # number of sites per unit cell
        self.positions = jnp.array(graph.positions)
        self.lattice_shape = tuple([int(e) for e  in graph.extent])
        if patch_shape is None:
            if self.n_sublattices == 1: #single-site unit cell 
                self.patch_shape = tuple([2]*self.lattice_dim)
            else:
                self.patch_shape = tuple([1]*self.lattice_dim)
        else:
            self.patch_shape = tuple(patch_shape)

        if self.output_dim == 1: #For e.g ViT 
            if self.lattice_dim == 1:
                self.extract_patches = Extract1Dto1D(
                    patch_size = self.patch_shape[0], 
                    lattice_size = self.lattice_shape[0],
                    nb = self.n_sublattices
                    )
            elif self.lattice_dim == 2:
                self.extract_patches = Extract2Dto1D(
                    patch_shape = self.patch_shape, 
                    lattice_shape = self.lattice_shape,
                    nb = self.n_sublattices,
                    )
            elif self.lattice_dim >= 3:
                self.extract_patches = Extract3Dto1D(
                    patch_shape = self.patch_shape, 
                    lattice_shape = self.lattice_shape,
                    nb = self.n_sublattices)

        elif self.output_dim == 2:
            if self.lattice_dim != 2:
                raise ValueError(
                    "Lattice must be two dimensional for two-dimensional patched lattice"
                )
            self.extract_patches = Extract2Dto2D(
                patch_shape = self.patch_shape, 
                lattice_shape = self.lattice_shape,
                nb = self.n_sublattices
                )
        else:
            raise NotImplementedError(
                "Patching for > 2 output dimensions not implemented yet"
            )
            
        self.plattice_shape = tuple(
            [int(self.lattice_shape[i] // self.patch_shape[i]) for i in range(self.lattice_dim)]
        )

        if self.output_dim == 1:
            self.compute_positions = PositionsTo1D(
                graph, extract_patches=self.extract_patches
            )
        elif self.output_dim == 2:
            self.compute_positions = PositionsTo2D(
                graph, extract_patches=self.extract_patches
            )
        else:
            raise NotImplementedError(
                "Patching for > 2 output dimensions not implemented yet"
            )

    def positions_to2d(self) -> Array:
        """
        Return the positions of the centre of patches in shape (npx,npy, 2).
        The order of patches is the same as that from extract_patches_to2d
        """
        patched_positions = self.extract_patches(
            self.positions.transpose()
        )  # (ndim,npx,npy,patch_size)
        patch_positions = jnp.mean(
            patched_positions, axis=-1
        )  # (ndim,npx,npy), gives centre of sites making up patch
        return patch_positions.transpose((1, 2, 0))  # (npx,npy,ndim)


class Reshape:
    """
    Reshape the (..., Nsites) input to (..., Nx, Ny) and define function returning positions of sites.
    Designed for netket 2D lattices with single-site unit cell
    """

    def __init__(self, graph: Lattice):
        assert graph.ndim == 2  # only for 2D lattices at the moment
        self.lattice_dim = graph.ndim
        self.lattice_shape = graph.extent
        self.graph = graph
        self.positions = jnp.array(
            graph.positions.reshape(self.lattice_shape[0], self.lattice_shape[1], 2)
        )

    def reshape(self, x: Array) -> Array:
        """
        Reshape the (..., Nsites) input to (..., Nx, Ny)
        """
        return x.reshape(*x.shape[:-1], self.lattice_shape[0], self.lattice_shape[1])

    def compute_positions(self) -> Array:
        """
        Return the positions of the sites in shape (Nx, Ny, 2).
        """
        return self.positions
