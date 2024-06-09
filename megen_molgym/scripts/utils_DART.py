import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import argparse, logging, string, random, sys, os, pdb
import warnings, shutil


device = "cuda:0"

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
        # pdb.set_trace()
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

class sep_ijkl_dataset_DART(Dataset):
    def __init__(self, data):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        # pdb.set_trace()
        self.data = data
        self.ener = self.data["ener"]
        batch = self.data["desc"]
        # self.ener = torch.tensor([j for i in self.ener for j in i], dtype=torch.float, device=device)
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

class calc(Dataset):       
    def __init__(self, energy, real_sph, pred_sph):
        super(calc, self).__init__()
        self.mask = self.get_mask(real_sph)
        self.energy = energy.reshape(-1,1).to(device)
#         pdb.set_trace()
        self.real_sph = real_sph[:self.mask]
        self.pred_sph = pred_sph[:self.mask]
        self.real_cart = self.sph2cart(self.real_sph[:self.mask])
        self.pred_cart = self.sph2cart(self.pred_sph[:self.mask])
        self.real_desc = self.Get_TAD(self.real_cart[:self.mask])
        self.pred_desc = self.Get_TAD(self.pred_cart[:self.mask])
    
    def get_mask(self, ai):
        ai_sum = ai.sum(axis=1)
        same_shape = ai_sum.shape
        ones = torch.ones(same_shape, device=device)
        zeros = torch.zeros(same_shape, device=device)
        make_zero = torch.where(ai_sum==0, zeros, ones)
        mask = int(make_zero.sum().item())
        return mask

    def sph2cart(self, sph):
        r, inc, azi = sph[:,0], sph[:,1], sph[:,2]
        x = r * torch.sin(inc) * torch.cos(azi)
        y = r * torch.sin(inc) * torch.sin(azi)
        z = r * torch.cos(inc)
        return torch.cat((x.unsqueeze(dim=1), y.unsqueeze(dim=1), z.unsqueeze(dim=1)), 1)

    def cart2sph(self, x, y, z):
        XsqPlusYsq = x**2 + y**2
        r = torch.sqrt(XsqPlusYsq + z**2)                      # r
        elev = torch.acos(z/r)                                 # theta
        az = torch.atan2(y,x)                                  # phi
        return torch.cat((r.unsqueeze(dim=1), elev.unsqueeze(dim=1), az.unsqueeze(dim=1)), 1)

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
        des_with_labels = torch.tensor([], device="cuda")
        num_atoms = cm.shape[0]
        for i in range(cm.shape[0]):
            tmp = torch.cat((cm, torch.zeros(num_atoms,1).to(device)), dim=1)  # add 4th with num_atoms and 5th col with all zeros
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
        cm = cm.to("cpu").numpy()
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
#         pdb.set_trace()
        self.dist_mat = torch.cdist(coord.reshape(-1,3), coord.reshape(-1,3))
        desc = self.rdf_like_desc()
        desc = self.add_labels_ijkl_L3(desc)
        desc = self.generate_sep_ijkl_L3(desc)
        atm_i = torch.tensor([i.reshape(3) for i in desc[0][0]]).to(device)
        atm_j = pad_sequence([torch.tensor(i) for i in desc[0][1]], batch_first=True).to(device)
        atm_k = pad_sequence([torch.tensor(i) for i in desc[0][2]], batch_first=True).to(device)
        atm_l = pad_sequence([torch.tensor(i) for i in desc[0][3]], batch_first=True).to(device)
#         return {"ener": self.energy, "atm_i":atm_i, "atm_j":atm_j, "atm_k":atm_k, "atm_l":atm_l, "desc":desc}
        return {"ener": self.energy, "atm_i":atm_i.unsqueeze(dim=0), "atm_j":atm_j.unsqueeze(dim=0), "atm_k":atm_k.unsqueeze(dim=0), "atm_l":atm_l.unsqueeze(dim=0)}


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        pass
        # print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


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