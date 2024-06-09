import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import argparse, logging, string, random, sys, os, pdb


device = "cuda"

class DART_Net(nn.Module):
    def __init__(self, lh, lo, li2, li3, li4):
        super(DART_Net, self).__init__()
        self.fi1 = nn.Linear(3, lh)
        self.fi2 = nn.Linear(lh, lo)
        
        self.fj1 = nn.Linear(3, lh)
        self.fj2 = nn.Linear(lh, lo)

        self.fk1 = nn.Linear(3, lh)
        self.fk2 = nn.Linear(lh, lo)
        
        self.fl1 = nn.Linear(3, lh)
        self.fl2 = nn.Linear(lh, lo)
        
        self.inter1 = nn.Linear(lo, li2)
        self.inter2 = nn.Linear(li2, li3)
        self.inter3 = nn.Linear(li3, li4)
        self.inter4 = nn.Linear(li4, 1)
        self.mask = nn.Linear(lo, lo, bias=False)
        
    def forward(self, ai, aj, ak, al):
        ai_sum = ai.sum(axis=2)
        same_shape = ai_sum.shape
        ones = torch.ones(same_shape, device=device)
        zeros = torch.zeros(same_shape, device=device)
        make_zero = torch.where(ai_sum==0, zeros, ones)
        ai_mask = make_zero.unsqueeze(dim=2)

        aj_sum = aj.sum(axis=3)
        same_shape = aj_sum.shape
        ones = torch.ones(same_shape, device=device)
        zeros = torch.zeros(same_shape, device=device)
        make_zero = torch.where(aj_sum==0, zeros, ones)
        aj_mask = make_zero.unsqueeze(dim=3)

        ak_sum = ak.sum(axis=3)
        same_shape = ak_sum.shape
        ones = torch.ones(same_shape, device=device)
        zeros = torch.zeros(same_shape, device=device)
        make_zero = torch.where(ak_sum==0, zeros, ones)
        ak_mask = make_zero.unsqueeze(dim=3)

        al_sum = al.sum(axis=3)
        same_shape = al_sum.shape
        ones = torch.ones(same_shape, device=device)
        zeros = torch.zeros(same_shape, device=device)
        make_zero = torch.where(al_sum==0, zeros, ones)
        al_mask = make_zero.unsqueeze(dim=3)

        ######### atom_i ############
        ai = F.celu(self.fi1(ai), 0.1)
        ai = F.celu(self.fi2(ai), 0.1)
        ai = ai * ai_mask
        
        ######### atom_j ############
        aj = F.celu(self.fj1(aj), 0.1)
        aj = F.celu(self.fj2(aj), 0.1)
        aj = aj * aj_mask

        ######### atom_k ############
        ak = F.celu(self.fk1(ak), 0.1)
        ak = F.celu(self.fk2(ak), 0.1)
        ak = ak * ak_mask
        
        ######### atom_l ############
        al = F.celu(self.fl1(al), 0.1)
        al = F.celu(self.fl2(al), 0.1)
        al = al * al_mask

        ########### interactions of i and j atoms ############
        atm = ai + aj.sum(axis=2) + ak.sum(axis=2) + al.sum(axis=2) # sum all interaction
        atm = F.celu(self.inter1(atm), 0.1)
        atm = F.celu(self.inter2(atm), 0.1)
        atm = F.celu(self.inter3(atm), 0.1)
        atm = self.inter4(atm)
        atm = atm * ai_mask
        return atm

class sep_ijkl_dataset(Dataset):
    def __init__(self, file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.data = np.load(file, allow_pickle=True)
        self.ener = self.data["ener"]
        batch = self.data["desc"]
        self.ener = torch.tensor([j for i in self.ener for j in i], dtype=torch.float, device=device)
        batch_size = len(self.ener)
        max_atoms = []

        for batch_idx in range(batch_size):
            lol = batch[batch_idx]
            max_atoms.append(len(lol[0])) # find longest sequence
            max_atoms.append(max([len(i) for i in lol[1]])) # find longest sequence
            max_atoms.append(max([len(i) for i in lol[2]])) # find longest sequence
            max_atoms.append(max([len(i) for i in lol[3]])) # find longest sequence
        iic = max_atoms[0::4]
        jjc = max_atoms[1::4]
        kkc = max_atoms[2::4]
        llc = max_atoms[3::4]

        des_j = []
        des_k = []
        des_l = []
        for i in range(batch_size):
            const_atom_count_i = max(iic) - iic[i]
            const_atom_count_j = max(jjc) - jjc[i]
            const_atom_count_k = max(kkc) - kkc[i]
            const_atom_count_l = max(llc) - llc[i]
            a_j = torch.zeros(const_atom_count_i, const_atom_count_j, 3)
            a_k = torch.zeros(const_atom_count_i, const_atom_count_k, 3)
            a_l = torch.zeros(const_atom_count_i, const_atom_count_l, 3)
            des_j.append(pad_sequence([torch.tensor(i) for i in batch[i][1]] + [i for i in a_j]))
            des_k.append(pad_sequence([torch.tensor(i) for i in batch[i][2]] + [i for i in a_k]))
            des_l.append(pad_sequence([torch.tensor(i) for i in batch[i][3]] + [i for i in a_l]))
        
        self.des_i = pad_sequence([torch.tensor(batch[i][0]) for i in range(batch_size)], batch_first=True).squeeze().float().to(device)
        des_j = pad_sequence(des_j, batch_first=True)
        self.des_j = torch.transpose(des_j, 1, 2).float().to(device)

        des_k = pad_sequence(des_k, batch_first=True)
        self.des_k = torch.transpose(des_k, 1, 2).float().to(device)

        des_l = pad_sequence(des_l, batch_first=True)
        self.des_l = torch.transpose(des_l, 1, 2).float().to(device)
        
    def __len__(self):
        return len(self.ener)
    
    def __getitem__(self, idx):
        sample = {"atm_i": self.des_i[idx], "atm_j": self.des_j[idx], "atm_k": self.des_k[idx], "atm_l": self.des_l[idx], "energy": self.ener[idx]}
        return sample


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint