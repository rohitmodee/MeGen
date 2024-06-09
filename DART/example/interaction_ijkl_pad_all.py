import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse, logging, string, random, sys, os, pdb
from datetime import datetime
import warnings, shutil
import pdb

#used for creating a "unique" id for a run (almost impossible to generate the same twice)
def id_generator(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

#define command line arguments
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument("--restart", type=str, default=None,  help="restart training from a specific folder")
parser.add_argument("--dataset", type=str,   help="file path to dataset")
parser.add_argument("--ga70_dataset", type=str,   help="file path to dataset")
parser.add_argument("--batch_size", type=int, help="batch size used per training step")
parser.add_argument("--adam_lr", default=0.0001, type=float, help="learning rate used by the optimizer")
parser.add_argument("--decay_steps", type=int, help="decay the learning rate with patience of N steps by decay_rate")
parser.add_argument("--max_epochs", type=int, help="Maximum number of epoch")
parser.add_argument("--seed", type=int, help="seed for split")
parser.add_argument("--lh", type=int, help="Number of hidden layers")
parser.add_argument("--lo", type=int, help="Number of outfut feature layers")
parser.add_argument("--li2", type=int, help="Number of interaction feature layers")
parser.add_argument("--li3", type=int, help="Number of interaction feature layers")
parser.add_argument("--li4", type=int, help="Number of interaction feature layers")
#if no command line arguments are present, config file is parsed
config_file='config.txt'
if len(sys.argv) == 1 or sys.argv[1] == "-f":
    if os.path.isfile(config_file):
        args = parser.parse_args(["@"+config_file])
    else:
        args = parser.parse_args(["--help"])
else:
    args = parser.parse_args()

#create directories
#a unique directory name is created for this run based on the input
if args.restart is None:
    directory="runs/" + datetime.now().strftime("%Y%m%d%H%M%S") + "_" + id_generator()
else:
    directory="runs/" + args.restart


if not os.path.exists(directory):
    os.makedirs(directory)

log_dir = os.path.join(directory, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#create log file
fname = directory + "/train"
logging.basicConfig(filename=fname+".log",level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True

#write config file (to restore command line arguments)
logging.info("writing args to file...")
with open(os.path.join(directory, config_file), 'w') as f:
    for arg in vars(args):
        f.write('--'+ arg + '='+ str(getattr(args, arg)) + "\n")

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class InteractionLayer(nn.Module):
    def __init__(self, lh, lo, li2, li3, li4):
        super(InteractionLayer, self).__init__()
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
#         self.m = nn.Dropout(p=0.3)
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
        # atm = torch.cat([self.mask(ai), aj.sum(axis=2), ak.sum(axis=2), al.sum(axis=2)], dim=2)
        # atm = self.mask(atm)
        atm = F.celu(self.inter1(atm), 0.1)
        atm = F.celu(self.inter2(atm), 0.1)
        atm = F.celu(self.inter3(atm), 0.1)
#         atm = atm + ai_res + aj_res.sum(axis=2) + ak_res.sum(axis=2) + al_res.sum(axis=2)
        atm = self.inter4(atm)
        atm = atm * ai_mask
        return atm

variable_model = InteractionLayer(args.lh, args.lo, args.li2, args.li3, args.li4).to(device)
logging.info(variable_model)

def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

variable_model.apply(init_params)

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

        try:
            self.coor = self.data["coor"]
            tmp_coor = []
            for i in range(len(self.coor)):
                offset = 94-self.coor[i].shape[1]
                tmp_pad = np.pad(self.coor[i], [(0,0),(0,offset), (0,0)], mode='constant')
                tmp_coor.append(tmp_pad)
            self.coor = np.array([j for i in tmp_coor for j in i])
        except KeyError:
            pass
        
    def __len__(self):
        return len(self.ener)
    
    def __getitem__(self, idx):
        try:
            sample = {"atm_i": self.des_i[idx], "atm_j": self.des_j[idx], "atm_k": self.des_k[idx], "atm_l": self.des_l[idx], "energy": self.ener[idx], "coor": self.coor[idx]}
        except AttributeError:    
            sample = {"atm_i": self.des_i[idx], "atm_j": self.des_j[idx], "atm_k": self.des_k[idx], "atm_l": self.des_l[idx], "energy": self.ener[idx]}
        return sample

# print(args.dataset)
logging.info("reading data")
desc_data = sep_ijkl_dataset(args.dataset)
# test_ga70 = sep_ijkl_dataset(args.ga70_dataset)

validation_split = .1
test_split = .1
shuffle_dataset = True
random_seed= args.seed

logging.info("Creating data indices for training and validation splits")
# Creating data indices for training and validation splits:
dataset_size = len(desc_data)
indices = list(range(dataset_size))
splitv = int(np.floor(validation_split * dataset_size))
splitt = int(np.floor(test_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices, test_indices = indices[splitt+splitv:], indices[:splitv], indices[splitv:splitt+splitv]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

logging.info("Creating dataloaders")
trainloader = DataLoader(desc_data, batch_size=args.batch_size, sampler=train_sampler)
validloader = DataLoader(desc_data, batch_size=args.batch_size, sampler=valid_sampler)
testloader = DataLoader(desc_data, batch_size=args.batch_size, sampler=test_sampler)
# ga70_loader = DataLoader(test_ga70, batch_size=args.batch_size, shuffle=False)

# criterion = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(variable_model.parameters(), lr=args.adam_lr)
# optimizer = optim.SGD(variable_model.parameters(), lr=0.0001, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=args.decay_steps, verbose=True, eps=1e-09)

epochal_train_losses = []
epochal_val_losses  = []
test_mae = []
num_epochs = args.max_epochs
epoch_freq = 1

def md_test(variable_model, ga70_loader):
    coor = torch.tensor([], device="cuda")
    pred_energy = torch.tensor([], device="cuda")
    cluster_size = torch.tensor([], device="cuda")
    variable_model.eval()
    with torch.no_grad():
        for batch in ga70_loader:
            energy = variable_model(batch["atm_i"], batch["atm_j"], batch["atm_k"], batch["atm_l"])
            energy = energy.sum(axis=1).squeeze()
            pred_energy = torch.cat((pred_energy, energy))
            cluster_size = torch.cat((cluster_size, batch["atm_i"][:,0].sum(axis=1)))
            coor = torch.cat((coor, batch["coor"].float().to(device)))
        # results = torch.stack((cluster_size, pred_energy, coor), axis=1)
        return pred_energy, cluster_size, coor

def test(variable_model, testloader):
    mae = torch.nn.L1Loss()
    rmse = torch.nn.MSELoss()
    pred_energy = torch.tensor([], device="cuda")
    real_energy = torch.tensor([], device="cuda")
    cluster_size = torch.tensor([], device="cuda")
    variable_model.eval()
    with torch.no_grad():
        for batch in testloader:
            energy = variable_model(batch["atm_i"], batch["atm_j"], batch["atm_k"], batch["atm_l"])
            energy = energy.sum(axis=1).squeeze()
            pred_energy = torch.cat((pred_energy, energy))
            real_energy = torch.cat((real_energy, batch["energy"]))
            cluster_size = torch.cat((cluster_size, batch["atm_i"][:,0].sum(axis=1)))
        results = torch.stack((cluster_size, real_energy, pred_energy), axis=1)
        test_loss = mae(pred_energy, real_energy)
        rmse_loss = torch.sqrt(rmse(pred_energy, real_energy))
        logging.info("Test MAE: {} and Test RMSE: {}".format(test_loss.item(), rmse_loss.item()))
        return results, test_loss, rmse_loss
            
def train(variable_model, optimizer, epochal_train_losses, criterion):
    train_loss = 0.00
    n = 0
    variable_model.train()
    for batch in trainloader:
        optimizer.zero_grad()
#         pdb.set_trace()
        energy = variable_model(batch["atm_i"], batch["atm_j"], batch["atm_k"], batch["atm_l"])
        energy = energy.sum(axis=1)
        batch_loss = criterion(energy, batch["energy"].unsqueeze(1))
        batch_loss.backward()
        optimizer.step()
        
        train_loss += batch_loss.detach().cpu()
        n += 1
    train_loss /= n
    epochal_train_losses.append(train_loss)

def train_and_evaluate(variable_model, optimizer, scheduler, criterion, start_epoch=1, restart=None):
    if restart:
        restore_path = os.path.join(log_dir + "/last.pth.tar")
        checkpoint = load_checkpoint(restore_path, variable_model, optimizer)
        start_epoch = checkpoint["epoch"]
        #logging.info("Restoring from {} current epoch is {}".format(restore_path, start_epoch))

    logging.info("starting training from epoch:{}".format(start_epoch))
    best_val = 100000.00
    early_stopping_learning_rate = 1.0E-8

    for epoch in range(1, num_epochs+1):
        learning_rate = optimizer.param_groups[0]['lr']
        if learning_rate < early_stopping_learning_rate:
            break

        ############ training #############
        train(variable_model, optimizer, epochal_train_losses, criterion)
        
        ############ validation #############
        n=0
        val_loss = 0.0
        variable_model.eval()
        for batch in validloader:
            energy = variable_model(batch["atm_i"], batch["atm_j"], batch["atm_k"], batch["atm_l"])
            energy = energy.sum(axis=1)
            batch_loss = criterion(energy, batch["energy"].unsqueeze(1))
            val_loss += batch_loss.detach().cpu()
            n += 1
        val_loss /= n
        epochal_val_losses.append(val_loss)
        scheduler.step(val_loss)
     
        is_best = val_loss <= best_val
        if epoch % epoch_freq == 0:
            print("Epoch: {: <5} Train: {: <20} Val: {: <20}".format(epoch, epochal_train_losses[-1], val_loss))
            logging.info("Epoch:{} Train:{} Val:{} lr: {}".format(epoch, epochal_train_losses[-1], val_loss, learning_rate))
                    
        if is_best:
            best_val = val_loss
            save_checkpoint({'epoch': epoch + 1,
                             'val_loss': val_loss,
                             'train_loss': epochal_train_losses[-1],
                             'state_dict': variable_model.state_dict(),
                             'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=log_dir)

logging.info("LET'S Train!!!")
train_and_evaluate(variable_model, optimizer, scheduler, criterion)
pdb.set_trace()
results, test_mae, test_rmse = test(variable_model, testloader)
# results70, test_mae70, test_rmse70 = md_test(variable_model, ga70_loader)
# pdb.set_trace()
# pred_energy, cluster_size, coor = md_test(variable_model, ga70_loader)
# pred_energy, cluster_size, coor = pred_energy.cpu().numpy(), cluster_size.cpu().numpy(), coor.cpu().numpy()

results = results[results[:,0].argsort()].cpu().numpy()
np.savez(directory + "/results.npz", results=results)

# np.savez(directory + "/results_md.npz", pred_energy=pred_energy, cluster_size=cluster_size, coor=coor)

plt.plot(np.arange(0, len(epochal_train_losses[10:]), 1), epochal_train_losses[10:], label='Training Loss')
plt.plot(np.arange(0, len(epochal_train_losses[10:]), 1), epochal_val_losses[10:], label='validation loss')
plt.title("Validation MAE: {:.2f} Test MAE: {:.2f}".format(np.min(epochal_val_losses), test_mae.item()))
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(directory + "/training_curve.png", bbox_inches='tight')
plt.clf()

plt.plot(np.arange(0, len(epochal_train_losses), 1), epochal_train_losses, label='Training Loss')
plt.plot(np.arange(0, len(epochal_train_losses), 1), epochal_val_losses, label='validation loss')
plt.title("Validation MAE: {:.2f} Test MAE: {:.2f}".format(np.min(epochal_val_losses), test_mae.item()))
plt.legend(loc='best')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(directory + "/training_curve1.png", bbox_inches='tight')
plt.clf()

plt.hist(abs(results[:,1]-results[:,2]), density=True)
plt.title("Probability distribution of AE, MAE: {:.2f}".format(test_mae.item()))
plt.xlabel("MAE")
plt.ylabel("Freq")
plt.savefig(directory + "/distribution_abs_error.png", bbox_inches='tight')
plt.clf()

# plt.hist(abs(results70[:,1]-results70[:,2]), density=True)
# plt.title("Probability distribution of AE, MAE: {:.2f}".format(test_mae70.item()))
# plt.xlabel("Ga 70 MAE")
# plt.ylabel("Freq")
# plt.savefig(directory + "/distribution_abs_error_ga70.png", bbox_inches='tight')
# plt.clf()
trainset = desc_data[train_indices]["atm_i"]
lol = trainset[:,0].sum(axis=1).cpu().numpy()
plt.title("Train set Cluster distribution, size = {}".format(len(lol)))
plt.xlabel("Cluster size")
plt.ylabel("Freq")
plt.hist(lol, bins=69, align='mid')
plt.savefig(directory + "/cluster_distribution_trainset.png", bbox_inches='tight')
plt.clf()

trainset = desc_data[test_indices]["atm_i"]
lol = trainset[:,0].sum(axis=1).cpu().numpy()
plt.title("Test set Cluster distribution, size = {}".format(len(lol)))
plt.xlabel("Cluster size")
plt.ylabel("Freq")
plt.hist(lol, bins=69, align='mid')
plt.savefig(directory + "/cluster_distribution_testset.png", bbox_inches='tight')
plt.clf()

trainset = desc_data[val_indices]["atm_i"]
lol = trainset[:,0].sum(axis=1).cpu().numpy()
plt.title("Validation set Cluster distribution, size = {}".format(len(lol)))
plt.hist(lol, bins=69, align='mid')
plt.xlabel("Cluster size")
plt.ylabel("Freq")
plt.savefig(directory + "/cluster_distribution_validationset.png", bbox_inches='tight')
plt.clf()

# cluster_size = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
cluster_size = range(5, 71)
some_res = np.split(results[:,1:], np.unique(results[:, 0], return_index=True)[1][1:])
diff = [abs(i[:,0]-i[:,1]).mean() for i in some_res]
plt.title("Cluster size vs MAE")
plt.bar(cluster_size, diff, label='Training Loss')
plt.xlabel("Cluster size")
plt.ylabel("MAE")
plt.savefig(directory + "/mea_for_each_cluster.png", bbox_inches='tight')
plt.clf()
