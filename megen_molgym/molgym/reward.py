import abc
import time
from typing import Tuple, Dict

import ase.data
import numpy as np
from ase import Atoms, Atom

# from molgym.calculator import Sparrow
import pdb, os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from scripts.utils_DART import load_checkpoint, save_checkpoint, DART_Net, sep_ijkl_dataset_DART

DART_model = DART_Net(128, 128, 256, 128, 32).to("cuda:0")
# restore_path = os.path.join("energy_prediction_model/20201216072707_Ij2TYXpT/logs/last.pth.tar")
restore_path = os.path.join("../../DART/example/runs/20210709173818_aPtIx8Nf/logs/last.pth.tar")
checkpoint = load_checkpoint(restore_path, DART_model)
start_epoch = checkpoint["epoch"]

       
class MolecularReward(abc.ABC):
    @abc.abstractmethod
    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        raise NotImplementedError

    @staticmethod
    def get_minimum_spin_multiplicity(atoms) -> int:
        return sum(ase.data.atomic_numbers[atom.symbol] for atom in atoms) % 2 + 1


class InteractionReward(MolecularReward):
    def __init__(self) -> None:
        # Due to some mysterious bug in Sparrow, calculations get slower and slower over time.
        # Therefore, we generate a new Sparrow object every time.
        # self.calculator = Sparrow('PM6')
        # DART_model = DART_Net(args.lh, args.lo, args.li2, args.li3, args.li4).to("cuda")
        # self.DART_model = DART_Net(128, 128, 256, 128, 32).to("cuda")
        # # restore_path = os.path.join("energy_prediction_model/20201216072707_Ij2TYXpT/logs/last.pth.tar")
        # self.restore_path = os.path.join("../../DART/example/runs/20210709173818_aPtIx8Nf/logs/last.pth.tar")
        # self.checkpoint = load_checkpoint(self.restore_path, self.DART_model)
        # self.start_epoch = self.checkpoint["epoch"]

        self.settings = {
            'molecular_charge': 0,
            'max_scf_iterations': 128,
            'unrestricted_calculation': 1,
        }

        self.atom_energies: Dict[str, float] = {}

    def rdf_like_desc(self):
        num_atoms = len(self.dist_mat)
        peaks = []
        descriptor = []
        cutoffs = [3.49, 6.3, 15]
        for cut in cutoffs:
            peak = self.dist_mat<cut
            peaks.append(peak.sum(dim=0).unsqueeze(dim=0))
        # removes count of atoms from previous cutoff as we have only < cut and not > previous cut
        descriptor = torch.cat((peaks[0], peaks[1]-peaks[0], peaks[2]-peaks[1]), dim=0).T
        return descriptor
    
    def add_labels_ijkl_L3(self, cm):
        des_with_labels = torch.tensor([], device="cpu")
        # des_with_labels = torch.tensor([], device="cuda")
        num_atoms = cm.shape[0]
        for i in range(cm.shape[0]):
            # tmp = torch.cat((cm, torch.zeros(num_atoms,1).to("cuda")), dim=1)  # add 4th with num_atoms and 5th col with all zeros
            tmp = torch.cat((cm, torch.zeros(num_atoms,1)), dim=1)  # add 4th with num_atoms and 5th col with all zeros
            l0 = torch.logical_and(self.dist_mat[i]<3.49, self.dist_mat[i]>0.5)  # cutoff for j
            l1 = torch.logical_and(self.dist_mat[i]>=3.49, self.dist_mat[i]<6.3) # cutoff for k
            l2 = self.dist_mat[i]>=6.3                                                # cutoff for l
            tmp[i,-1] = 1                           # add label of i atom as 1
            tmp[l0[:num_atoms],3] = 2               # add label of j atom as 2
            tmp[l1[:num_atoms],3] = 3               # add label of k atom as 3
            tmp[l2[:num_atoms],3] = 4               # add label of l atom as 4
            des_with_labels = torch.cat((des_with_labels, tmp.reshape(-1)))
        return des_with_labels.reshape(num_atoms,-1)
    
    def generate_sep_ijkl_L3(self, cm):
        cm = cm.numpy()
        descriptor = []
        num_atoms = len(cm)
        single_mol = cm.reshape(num_atoms,num_atoms,-1)
        # separate i, j, k and l atoms based on labels and removes labels
        i_atoms = [single_mol[i, single_mol[i,:,3]==1][:,:3] for i in range(num_atoms)]
        j_atoms = [single_mol[i, single_mol[i,:,3]==2][:,:3] for i in range(num_atoms)]
        k_atoms = [single_mol[i, single_mol[i,:,3]==3][:,:3] for i in range(num_atoms)]
        l_atoms = [single_mol[i, single_mol[i,:,3]==4][:,:3] for i in range(num_atoms)]
        combine_ijkl = [i_atoms, j_atoms, k_atoms, l_atoms]
        descriptor.append(combine_ijkl)
        return np.array(descriptor, dtype=object)
    
    def Get_TAD(self, coord):
        coord = torch.from_numpy(coord)
        self.dist_mat = torch.cdist(coord.reshape(-1,3), coord.reshape(-1,3))
        desc = self.rdf_like_desc()
        desc = self.add_labels_ijkl_L3(desc)
        desc = self.generate_sep_ijkl_L3(desc)
        atm_i = torch.tensor([i.reshape(3) for i in desc[0][0]]).to("cuda:0")
        atm_j = pad_sequence([torch.tensor(i) for i in desc[0][1]], batch_first=True).to("cuda:0")
        atm_k = pad_sequence([torch.tensor(i) for i in desc[0][2]], batch_first=True).to("cuda:0")
        atm_l = pad_sequence([torch.tensor(i) for i in desc[0][3]], batch_first=True).to("cuda:0")
#         return {"ener": self.energy, "atm_i":atm_i, "atm_j":atm_j, "atm_k":atm_k, "atm_l":atm_l, "desc":desc}
        return {"atm_i":atm_i.unsqueeze(dim=0), "atm_j":atm_j.unsqueeze(dim=0), "atm_k":atm_k.unsqueeze(dim=0), "atm_l":atm_l.unsqueeze(dim=0)}

    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        start = time.time()
        # self.calculator = Sparrow('PM6')

        all_atoms = atoms.copy()
        all_atoms.append(new_atom)

        desc = self.Get_TAD(all_atoms.positions)

        # e_tot = self._calculate_energy(all_atoms)
        # e_parts = self._calculate_energy(atoms) + self._calculate_atomic_energy(new_atom)
        # delta_e = e_tot - e_parts
        # pdb.set_trace()
        delta_e = DART_model(desc["atm_i"], desc["atm_j"], desc["atm_k"], desc["atm_l"])
        print("-->", delta_e, desc["atm_i"].shape)
        delta_e = delta_e.sum(axis=1).item() * 0.036749305           # eV to Hartree
        elapsed = time.time() - start

        reward = -1 * delta_e
        # time.sleep(1)
        info = {
            'elapsed_time': elapsed,
        }

        return reward, info

    def _calculate_atomic_energy(self, atom: Atom) -> float:
        if atom.symbol not in self.atom_energies:
            atoms = Atoms()
            atoms.append(atom)
            self.atom_energies[atom.symbol] = self._calculate_energy(atoms)
        return self.atom_energies[atom.symbol]

    def _calculate_energy(self, atoms: Atoms) -> float:
        if len(atoms) == 0:
            return 0.0

        self.calculator.set_elements(list(atoms.symbols))
        self.calculator.set_positions(atoms.positions)
        self.settings['spin_multiplicity'] = self.get_minimum_spin_multiplicity(atoms)
        self.calculator.set_settings(self.settings)
        return self.calculator.calculate_energy()


class SolvationReward(InteractionReward):
    def __init__(self, distance_penalty=0.01) -> None:
        super().__init__()

        self.distance_penalty = distance_penalty

    def invalid_action(self, existing_atom: Atoms, new_atom: Atom):
        cut_1nn = [2.3, 3.1]
        cut_2nn = [2.5, 3.0]
        cut_3nn = [2.5, 5.5]

        is_valid_action = True
        if len(existing_atom.positions) > 2:
            nn_dist = np.linalg.norm(existing_atom.positions - new_atom.position, axis=1)
            sorted_nn_dist = np.sort(nn_dist)[:4]
            if sorted_nn_dist[0] < cut_1nn[0] or sorted_nn_dist[0] > cut_1nn[1]:
                is_valid_action = False
            if sorted_nn_dist[1] < cut_2nn[0] or sorted_nn_dist[1] > cut_2nn[1]:
                is_valid_action = False
            if sorted_nn_dist[2] < cut_3nn[0] or sorted_nn_dist[2] > cut_3nn[1]:
                is_valid_action = False
            print(nn_dist, is_valid_action)
        # if np.linalg.norm(existing_atom.position - new_atom.position) < :
        return is_valid_action

    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        start_time = time.time()
        self.calculator = Sparrow('PM3')

        all_atoms = atoms.copy()
        all_atoms.append(new_atom)

        # if len()
        is_valid_action = self.invalid_action(atoms, new_atom)
        if is_valid_action:
            invalid_action_penalty = 0
        else:
            invalid_action_penalty = -10

        e_tot = self._calculate_energy(all_atoms)
        e_parts = self._calculate_energy(atoms) + self._calculate_atomic_energy(new_atom)
        delta_e = e_tot - e_parts

        distance = np.linalg.norm(new_atom.position)

        reward = -1 * (delta_e + self.distance_penalty * distance) + invalid_action_penalty
        print(f"reward {reward}")
        info = {
            'elapsed_time': time.time() - start_time,
        }

        return reward, info

class CustomReward(InteractionReward):
    def __init__(self, distance_penalty=0.1) -> None:
        super().__init__()

        self.distance_penalty = distance_penalty

    def invalid_action(self, existing_atom: Atoms, new_atom: Atom):
        cut_1nn = [2.3, 3.1]
        cut_2nn = [2.5, 3.0]
        cut_3nn = [2.5, 5.5]

        is_valid_action = True
        if len(existing_atom.positions) > 2:
            nn_dist = np.linalg.norm(existing_atom.positions - new_atom.position, axis=1)
            sorted_nn_dist = np.sort(nn_dist)[:4]
            if sorted_nn_dist[0] < cut_1nn[0] or sorted_nn_dist[0] > cut_1nn[1]:
                is_valid_action = False
            if sorted_nn_dist[1] < cut_2nn[0] or sorted_nn_dist[1] > cut_2nn[1]:
                is_valid_action = False
            if sorted_nn_dist[2] < cut_3nn[0] or sorted_nn_dist[2] > cut_3nn[1]:
                is_valid_action = False
            # print(nn_dist, is_valid_action)
        # if np.linalg.norm(existing_atom.position - new_atom.position) < :
        return is_valid_action

    def calculate(self, atoms: Atoms, new_atom: Atom) -> Tuple[float, dict]:
        # print("custom reward called")
        start_time = time.time()
        # self.calculator = Sparrow('PM6')

        all_atoms = atoms.copy()
        all_atoms.append(new_atom)

        # if len()
        is_valid_action = self.invalid_action(atoms, new_atom)
        if is_valid_action:
            invalid_action_penalty = 0
        else:
            invalid_action_penalty = -10


        desc = self.Get_TAD(all_atoms.positions)

        # e_tot = self._calculate_energy(all_atoms)
        # e_parts = self._calculate_energy(atoms) + self._calculate_atomic_energy(new_atom)
        # delta_e = e_tot - e_parts
        # pdb.set_trace()
        delta_e = DART_model(desc["atm_i"], desc["atm_j"], desc["atm_k"], desc["atm_l"])
        # print("-->", delta_e, desc["atm_i"].shape)
        # delta_e = delta_e.sum(axis=1).item() * 0.036749305           # eV to Hartree
        delta_e = delta_e.sum(axis=1).item() * 0.001593601097421           # kcal/mol to Hartree

        # distance = np.linalg.norm(new_atom.position)
        CoM = np.mean(atoms.positions, axis=0)
        distance = np.linalg.norm(CoM-new_atom.position)

        reward = -1 * (delta_e + self.distance_penalty * distance) + invalid_action_penalty
        # reward = -1 * (delta_e + self.distance_penalty * distance) #+ invalid_action_penalty
        # print(f"size {len(all_atoms.positions)} reward {reward}")
        info = {
            'elapsed_time': time.time() - start_time,
        }
        return reward, info