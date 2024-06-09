import numpy as np
import pdb

root_dir = "runs/"
run_dir = "20210227083740_mvJ5jVYL"
data = np.load(root_dir + run_dir + "/results_md.npz")
write_file = root_dir + run_dir + "/results_md_text.dat"

def remove_padding(data):
	new_coord = []
	new_shape = []
	for coord in data["coor"]:
		mask = np.sum(coord, axis=1)!=0
		new_coord.append(coord[mask])
		new_shape.append(new_coord[-1].shape[0])
	return new_coord, new_shape

def coor_to_str(coord):
	coor_ss = ""
	for i in coord:
		coor_ss += "Ga\t"
		coor_ss += "\t".join([str(j) for j in i]) + "\n"
	return coor_ss

coor, shape = remove_padding(data)

with open(write_file, "w") as f:
	for i in range(len(coor)):
		# pdb.set_trace()
		cluster_size = str(shape[i]) + "\n"
		energy = str(data["pred_energy"][i]) + "\n"
		coor_str = coor_to_str(coor[i])
		all_str = cluster_size + energy + coor_str + "\n\n"
		f.write(all_str)
		break

