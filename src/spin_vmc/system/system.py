# 2D spin models
import netket as nk
import numpy as np
import argparse
import warnings
import einops

from collections.abc import Sequence
from spin_vmc.system.base import SpinSystem, SpinonSystem, BaseSystem
from spin_vmc.system.utils import reflect_and_translate, reflect_and_translate_group, is_symmetric
from netket.utils.group._point_group import PointGroup
from netket.utils.group import PermutationGroup, Identity
from netket.graph.space_group import SpaceGroup, Translation
from netket.utils.types import Array
from netket.nn.blocks import SymmExpSum
from nqs_nets.blocks.sign import SignNet, SignHelper, SignRule, DoubleSignNet
from nqs_nets.blocks import FlipExpSum
from netket.operator import LocalOperator
import nqs_support_core as nkp
from functools import partial
from nqs_support_core.utils.group.translations import translation_group_from_axis_translations
from typing import Optional
import jax.numpy as jnp


class SquareHeisenberg(SpinSystem):
    rotation_group = nk.utils.group.planar.C(4)
    sign_sublattices = (0, 2)  # sublattices for sign rule, assuming 2x2 patches

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--L", type=int, required=True, help="Linear size of the square lattice"
        )
        parser.add_argument(
            "--J", type=float, action="append", required=True, help="List of J values"
        )
        parser.add_argument(
            "--sign_rule",
            type=int,
            action="append",
            required=True,
            help="List of boolean sign rules for each J value",
        )
        parser.add_argument(
            "--patching",
            type=int,
            required=True,
            help="Whether using patching for network",
        )

    @staticmethod
    def read_arguments(args: argparse.Namespace):
        return args.L, args.J, [bool(i) for i in args.sign_rule], args.patching

    @staticmethod
    def reshape_xy(x: Array, lattice_shape: tuple) -> Array:
        """
        Reshape a (nbatch, nsites) array into an (nbatch,x,y,1) array, which can then have a 2d convolutional layer applied, where (x,y) label the real space coordinates of the point
        """
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))  # add batch dimension
        x = x.reshape(
            (x.shape[0], lattice_shape[0], lattice_shape[1])
        )  # (nbatch, x, y)
        x = x.reshape(x.shape + (1,))  # shape (nbatch, x, y, 1)
        return x

    def __init__(
        self,
        L: int,
        J: Sequence[float],
        sign_rule: Sequence[bool] = [False],
        patching: bool = True,
        sz_sector=0,
    ):
        super().__init__(N=int(L**2), sz_sector=sz_sector)
        self.J = J
        self.L = L
        self.name = "SquareHeisenberg"
        self.patching = patching
        self.graph = nk.graph.Square(length=L, max_neighbor_order=len(J), pbc=self.pbc)
        self.graph_name = "Square"
        # Get all the symmetries we will use to symmetrize the wavefunction
        self.graph_symmetries = {
            "C4": self.graph.point_group(self.rotation_group),
            "Full point group": self.graph.point_group(),
        }
        # Get translation symmetries if patching
        if patching:
            spacegroupbuilder = SpaceGroup(
                self.graph, nk.utils.group.trivial_point_group(ndim=2)
            )
            trans_x = spacegroupbuilder.translation_group(0)[1]  # translation +x
            trans_y = spacegroupbuilder.translation_group(1)[1]  # translation +y
            trans_xy = trans_x @ trans_y  # translation +x+y
            translation_group = PermutationGroup(
                [Identity(), trans_x, trans_y, trans_xy], degree=self.graph.n_nodes
            )
            self.graph_symmetries.update(
                {
                    "Translation": translation_group,
                    "T@C4": translation_group
                    @ self.graph_symmetries[
                        "C4"
                    ],  # group consisting of rotations and translations
                    "T@Full": translation_group
                    @ self.graph_symmetries[
                        "Full point group"
                    ],  # Full point group and translations
                }
            )

        if (
            not patching
        ):  # unsymmetrized + unsymmetrized + C4 + full point + spin parity
            self.symmetrizing_functions = (
                lambda net: net,
                lambda net: net,
                lambda net: SymmExpSum(net, self.graph_symmetries["C4"]),
                lambda net: SymmExpSum(net, self.graph_symmetries["Full point group"]),
                lambda net: FlipExpSum(
                    SymmExpSum(net, self.graph_symmetries["Full point group"])
                ),
            )
        else:  # unsymmetrized + translations + C4 + full point + spin parity
            self.symmetrizing_functions = (
                lambda net: net,
                lambda net: SymmExpSum(net, self.graph_symmetries["Translation"]),
                lambda net: SymmExpSum(net, self.graph_symmetries["T@C4"]),
                lambda net: SymmExpSum(net, self.graph_symmetries["T@Full"]),
                lambda net: FlipExpSum(
                    SymmExpSum(net, self.graph_symmetries["T@Full"])
                ),
            )

        if len(sign_rule) != len(J):
            warnings.warn(
                "len(sign_rule) and len(J) mismatched, increasing length of sign_rule by repeating first value..."
            )
            sign_rule = len(J) * [sign_rule[0]]
        self.sign_rule = sign_rule
        self.hamiltonian = nk.operator.Heisenberg(
            hilbert=self.hilbert_space,
            graph=self.graph,
            J=self.J,
            sign_rule=self.sign_rule,
        )
        self.hamiltonian_name = "Heisenberg"
        self.sampler_t = partial(nk.sampler.MetropolisExchange, graph=self.graph)

    def name_and_arguments_to_dict(self):
        return {
            "name": self.name,
            "L": self.L,
            "J": self.J,
            "sign_rule": self.sign_rule,
            "patching": self.patching,
        }


class ShastrySutherland(SpinSystem):
    basis_vecs = np.array([[2, 0], [0, 2]])
    unit_cell = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    custom_edges = [
        (0, 1, [1, 0], 0),
        (0, 1, [-1, 0], 0),
        (0, 2, [0, 1], 0),
        (0, 2, [0, -1], 0),
        (1, 3, [0, 1], 0),
        (1, 3, [0, -1], 0),
        (2, 3, [1, 0], 0),
        (2, 3, [-1, 0], 0),
        (0, 3, [-1, 1], 1),
        (2, 1, [1, 1], 1),
    ]
    rotation_group = nk.utils.group.planar.C(4)
    reflection_group = reflect_and_translate_group(
        45, np.array([1, -1])
    )  # {I, sigma_xy}
    reflection = PointGroup(
        [reflect_and_translate(45, np.array([1, -1]))], ndim=2
    )  # sigma_xy
    point_group = nk.utils.group._point_group.product(
        rotation_group, reflection_group
    )  # C4 x {I, sigma_xy}, full point group length 8
    glide_group = nk.utils.group._point_group.product(
        rotation_group, reflection
    )  # C4 x sigma_xy, glide group, reflections*rotations, length 4
    sign_sublattices = (0, 2)  # sublattices for sign rule, assuming 2x2 patches

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--L",
            type=int,
            required=True,
            help="Linear size of the underlying square lattice",
        )
        parser.add_argument(
            "--J", type=float, action="append", required=True, help="List of J values" #doesnt work well with command line specification
        )

    @staticmethod
    def read_arguments(args: argparse.Namespace):
        return args.L, args.J

    @staticmethod
    def get_graph(L: int) -> nk.graph.Graph:
        return nk.graph.Lattice(
            basis_vectors=ShastrySutherland.basis_vecs,
            site_offsets=ShastrySutherland.unit_cell,
            custom_edges=ShastrySutherland.custom_edges,
            extent=[L // 2, L // 2],
            pbc=True,
            point_group=ShastrySutherland.point_group,
        )

    @staticmethod
    def reshape_xy(x: Array, lattice_shape: tuple) -> Array:
        """
        Reshape a (nbatch, nsites) array into an (nbatch,x,y,1) array, which can then have a 2d convolutional layer applied, where (x,y) label the real space coordinates of the point
        """
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))  # add batch dimension
        x = ShastrySutherland.extract_patches_as2d(
            x=x, b=2, lattice_shape=lattice_shape
        )  # -> (nbatch, ux, uy, 4) where ux,uy label unit cell coords
        x = x.reshape(
            (x.shape[0], lattice_shape[0] // 2, lattice_shape[1] // 2, 2, 2)
        )  # (nbatch, ux,uy, dy, dx) , where dx, dy are positions within the unit cell ux,uy
        x = einops.rearrange(
            x, "batch ux uy dy dx -> batch (ux dx) (uy dy)"
        )  # shape (nbatch, x, y)
        x = x.reshape(x.shape + (1,))  # shape (nbatch, x, y, 1)
        return x

    def __init__(self, L: int, J: Sequence[float], sz_sector=0):
        if len(J) != 2:
            raise ValueError("Shastry-Sutherland model requires J1 and J2")
        super().__init__(N=int(L**2), sz_sector=sz_sector)
        self.J = J
        self.L = L
        self.name = "ShastrySutherland"
        self.graph = nk.graph.Lattice(
            basis_vectors=self.basis_vecs,
            site_offsets=self.unit_cell,
            custom_edges=self.custom_edges,
            extent=[L // 2, L // 2],
            pbc=self.pbc,
            point_group=self.point_group,
        )
        self.graph_name = "ShastrySutherland"
        self.graph_symmetries = {
            "C4": self.graph.point_group(self.rotation_group),
            "Glides": self.graph.point_group(self.glide_group),
            "Full point group": self.graph.point_group(),
        }
        # Define the symmetrizing_functions used in a symmetry ramping optimization
        self.symmetrizing_functions = (
            lambda net: net,  # unsymmetrized
            lambda net: SymmExpSum(net, self.graph_symmetries["C4"]),  # rotations
            lambda net: SymmExpSum(
                net, self.graph_symmetries["Full point group"]
            ),  # rotations and glides = full point group
            lambda net: FlipExpSum(
                SymmExpSum(net, self.graph_symmetries["Full point group"])
            ),  # S^z parity and full point group
        )

        self.hamiltonian = nk.operator.Heisenberg(
            hilbert=self.hilbert_space, graph=self.graph, J=self.J
        )
        self.hamiltonian_name = "Heisenberg"
        self.sampler_t = partial(nk.sampler.MetropolisExchange, graph=self.graph)

    def name_and_arguments_to_dict(self):
        return {
            "name": self.name,
            "L": self.L,
            "J": self.J,
            "sz_sector": self.sz_sector,
        }


