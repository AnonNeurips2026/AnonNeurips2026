import netket as nk
import json
import nqs_support_core as nkp

class BaseSystem:
    name = "None"
    graph_name = "None"
    hamiltonian_name = "None"
    pbc = True

    @staticmethod
    def argument_saver(
        arg_dict: dict, file_name: str, prefix: str = None, write_mode: str = "a"
    ):
        """
        Save the arg_dict to file fname
        """
        if not prefix == None:
            save_dict = {prefix: arg_dict}
        else:
            save_dict = arg_dict

        with open(file_name, write_mode) as f:
            json.dump(save_dict, f)

    def save(self, file_name: str, prefix: str = None, write_mode: str = "a"):
        """
        Save the system, loaded in by system.system.load
        """
        arg_dict = self.name_and_arguments_to_dict()
        self.argument_saver(arg_dict, file_name, prefix, write_mode)

    @staticmethod
    def argument_loader(file_name: str, prefix: str = None):
        """
        Load in the dictionary of arguments for system.system.from_dict
        """
        with open(file_name, "r") as f:
            load_dict = json.load(f)

        if not prefix == None:
            out_dict = load_dict[prefix]
        else:
            out_dict = load_dict

        return out_dict


class SpinSystem(BaseSystem):
    def __init__(self, N: int, local_spin=0.5, sz_sector=0):
        # From netket 3.14, inverted ordering must be specified to keep the old
        # behaviour of +1 == spin down and -1 == spin up. In the future, the
        # default will be flipped.
        self.local_spin = local_spin
        self.hilbert_space = nk.hilbert.Spin(
            s=self.local_spin, N=N, total_sz=sz_sector, inverted_ordering=True
        )
        self.N = N
        self.sz_sector = sz_sector

    def name_and_arguments_to_dict(self):
        """
        Convert the arguments for __init__ to a dict.
        Saving and then loading in this dict allows one to obtain the correct system from file.
        """
        return {
            "N": self.N,
            "local_spin": self.local_spin,
            "sz_sector": self.sz_sector,
        }


class SpinonSystem(BaseSystem):
    def __init__(self, n_orbitals: int):
        self.n_orbitals = n_orbitals
        self.n_fermions_per_spin = (n_orbitals // 2, n_orbitals // 2)
        self.s = 1 / 2
        constraint = nkp.hilbert.constraint.SpinHalfConstraint()
        self.hilbert_space = nk.hilbert.SpinOrbitalFermions(
            n_orbitals=self.n_orbitals,
            s=self.s,
            n_fermions_per_spin=self.n_fermions_per_spin,
            constraint=constraint,
        )

    def name_and_arguments_to_dict(self):
        """
        Convert the arguments for __init__ to a dict.
        Saving and then loading in this dict allows one to obtain the correct system from file.
        """
        return {
            "n_orbitals": self.n_orbitals,
        }
