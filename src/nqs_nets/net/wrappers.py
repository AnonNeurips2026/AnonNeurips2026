"""Slim ViT wrapper used by the paper experiments."""

from __future__ import annotations

import argparse

from nqs_nets.blocks.patching import Patching
from nqs_nets.net import ViT
from nqs_nets.net.base_wrapper import NetBase


class ViTNd(NetBase):
    """Adapter that builds the N-dimensional ViT ansatz used in the experiments."""

    nets = {
        "Vanilla": ViT.VanillaN,
        "FT": ViT.FT,
        "Positive": ViT.Positive,
        "FTReal": ViT.FTReal,
    }

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument("--depth", type=int, required=True)
        parser.add_argument("--d_model", type=int, required=True)
        parser.add_argument("--heads", type=int, required=True)
        parser.add_argument("--output_head", type=str, required=True)
        parser.add_argument("--expansion_factor", type=int, required=True)
        parser.add_argument("--q", type=float, action="append", required=False)
        parser.add_argument("--kernel_shape", type=int, action="append", required=False)
        parser.add_argument("--patch_shape", type=int, action="append", required=False)
        parser.add_argument("--sign_net", type=int, required=False, default=0)

    @staticmethod
    def read_arguments(args: argparse.Namespace):
        return {
            "depth": args.depth,
            "d_model": args.d_model,
            "heads": args.heads,
            "output_head": args.output_head,
            "expansion_factor": args.expansion_factor,
            "q": args.q,
            "kernel_shape": args.kernel_shape,
            "patch_shape": args.patch_shape,
            "sign_net": bool(args.sign_net),
        }

    def __init__(
        self,
        depth: int,
        d_model: int,
        heads: int,
        output_head: str,
        expansion_factor: int,
        system,
        q: tuple[float, ...] | None = (0, 0),
        kernel_shape: tuple[int, ...] | None = None,
        patch_shape: tuple[int, ...] | None = None,
        sign_net: bool = False,
        gutzwiller: bool = False,
    ):
        if gutzwiller:
            raise NotImplementedError("The anonymous package includes only the ViT ansatz.")
        if output_head not in self.nets:
            raise ValueError(f"Unknown ViT output head: {output_head}")

        self.name = "ViTNd"
        self.patch_shape = patch_shape
        self.depth = depth
        self.d_model = d_model
        self.heads = heads
        self.output_head = output_head
        self.expansion_factor = expansion_factor
        self.kernel_shape = None if kernel_shape is None else tuple(kernel_shape)
        self.q = None if q is None else tuple(q)
        self.sign_net = bool(sign_net)
        self.gutzwiller = False

        patches = Patching(system.graph, output_dim=1, patch_shape=patch_shape)
        self.patches = patches

        common_kwargs = dict(
            num_layers=depth,
            d_model=d_model,
            heads=heads,
            plattice_shape=patches.plattice_shape,
            extract_patches=patches.extract_patches,
            expansion_factor=expansion_factor,
            kernel_shape=self.kernel_shape,
            transl_invariant=True,
        )

        if output_head in {"FT", "FTReal"}:
            if self.q is None or len(self.q) != system.graph.ndim:
                raise ValueError("Fourier-transform ViT heads require one q value per graph dimension.")
            common_kwargs["q"] = self.q
            common_kwargs["compute_positions"] = patches.compute_positions

        self.network = self.nets[output_head](**common_kwargs)
        if self.sign_net:
            self.network = system.sign_net(self.network)

    def name_and_arguments_to_dict(self):
        return {
            "name": self.name,
            "depth": self.depth,
            "d_model": self.d_model,
            "heads": self.heads,
            "output_head": self.output_head,
            "expansion_factor": self.expansion_factor,
            "q": self.q,
            "kernel_shape": self.kernel_shape,
            "patch_shape": self.patch_shape,
            "sign_net": self.sign_net,
            "gutzwiller": self.gutzwiller,
        }


networks = {"ViTNd": ViTNd}


def from_dict(arg_dict: dict, system, network_name: str = "ViTNd"):
    arg_dict = dict(arg_dict)
    network = networks.get(str(arg_dict.pop("name", network_name)))
    if network is None:
        raise ValueError("Only ViTNd checkpoints are supported by this anonymous package.")
    return network(**arg_dict, system=system)


def load(file_name: str, system, prefix: str | None = None):
    return from_dict(NetBase.argument_loader(file_name, prefix), system)
