# MeGen - generation of gallium metal clusters using reinforcement learning

<img src="https://content.cld.iop.org/journals/2632-2153/4/2/025032/revision2/mlstacdc03f1_lr.jpg" width="70%">
<!-- ![Overview](https://content.cld.iop.org/journals/2632-2153/4/2/025032/revision2/mlstacdc03f1_lr.jpg) -->

<!-- ![new_workflow](https://content.cld.iop.org/journals/2632-2153/4/2/025032/revision2/mlstacdc03f3_lr.jpg) -->

In this work, we propose a 3D structure generator model named as MeGen (Metal cluster structure Generation). MeGen generates energetically low-lying 3D structures ofGa clusters in Cartesian coordinates using RL. Our model exploits the rotationally covariant state representation for 3D structure generation. We integrate this state representation into an actor-critic neural network architecture with a rotationally covariant auto-regressive policy, where the position of the atom to be placed is modeled through a flexible distribution based on spherical harmonics. The reward function is based on a fundamental physical property (energy). We use the previously developed machine learning (ML) model DART to calculate the energies of various conformers/isomers generated during training. Contrary to other models that use structural information, our DART model uses features that capture topological/connectivity information to predict metallic clusters energies. As the descriptor consists of just the atom counts in each shell, it captures topological information and is known as topological atomic descriptor. DART model provides accuracy comparable to DFT at minimal computational cost. In a nutshell, MeGen is an RL-based algorithm that generates low-energy 3D structures ofmetal clusters with biased structure generation towards low-energy structures. MeGen employs DART-predicted energy to evaluate the generated structures. Including DART reduces the number of compute-intensive optimizations and adds a bias in the model’s learning by rewarding low-energy generated structures. This makes MeGen more efficient than algorithms like basin hopping, random search, and genetic evolutions that do not have such a bias and require local structure optimization. Our

This repository allows to train reinforcement learning policies for designing gallium clusters directly in Cartesian coordinates. The agent builds gallium cluster of size "N+1" by taking _Ga_ atoms from a given _bag_ and placing them onto a 3D _canvas_. The 3D _canvas_ has seed structure of gallium cluster of size "N".

We have adapted the code from **MolGym**. We thank the authors for making the code available on github. MolGym package allows to train reinforcement learning policies for designing molecules directly in Cartesian coordinates. So MolGym was developed for organic molecules. We modified the code to work with clusters. In our case we focus on gallium clusters.

MeGen uses DART model as a reward function. The snippet *reward = CustomReward()* calls the DART model to calculate reward.


<!-- <img src="resources/intro.png" width="40%"> -->

## Setup

Dependencies:
* Python >= 3.7
* torch >= 1.5.1
* [MolGym](https://github.com/gncs/molgym)

Install required packages and library itself:
```
cd megen_molgym
pip install -r requirements.txt
pip install -e .
```

## Usage

You can use this code to train and evaluate reinforcement learning agents for generation of low-energy 3D gallium clusters. *data* directory contains the training data. DART directory contains trained DART model.

### Training
To perform the gallium (Ga18) structure generation experiment, run the following
```shell
cd scripts
python3 run_megen.py
```
Hyper-parameters and other details can be found in the paper.

### Evaluation

To generate learning curves, run the following command:
```shell
python3 plot.py --dir=results
```
Running this script will automatically generate a figure of the learning curve.

To write out the generated structures, run the following command:
```shell
python3 structures.py --dir=data --symbols=X,Ga
```
This should generate file named *structures_eval.xyz* which contains generated structures.

For post-processing and to write the structures in file *structures_eval.xyz* to POSCAR for duplicates removal and minimization using VASP use following command:
```shell
python3 read_gen_eval_stru.py
```


You can visualize the structures in the generated XYZ file using, for example, [ASE-GUI]https://wiki.fysik.dtu.dk/ase/ase/gui/gui.html#index-0).

## Citation

If you use this code, please cite our papers:
```txt
@article{megen_modee,
author = {Modee, Rohit and Verma, Ashwini and Joshi, Kavita and Priyakumar, U. Deva},
doi = {10.1088/2632-2153/acdc03},
issn = {2632-2153},
journal = {Mach. Learn. Sci. Technol.},
month = {jun},
number = {2},
pages = {025032},
title = {{MeGen - generation of gallium metal clusters using reinforcement learning}},
url = {http://iopscience.iop.org/article/10.1088/2632-2153/acdc03},
volume = {4},
year = {2023}
}
@article{Modee2021,
author = {Modee, Rohit and Agarwal, Sheena and Verma, Ashwini and Joshi, Kavita and Priyakumar, U. Deva},
doi = {10.1039/d1cp02956h},
issn = {14639076},
journal = {Phys. Chem. Chem. Phys.},
number = {38},
pages = {21995--22003},
title = {{DART: Deep learning enabled topological interaction model for energy prediction of metal clusters and its application in identifying unique low energy isomers}},
url = {http://xlink.rsc.org/?DOI=D1CP02956H},
volume = {23},
year = {2021}
}
```

MolGym papers:
```txt
@inproceedings{Simm2020Reinforcement,
  title = {Reinforcement Learning for Molecular Design Guided by Quantum Mechanics},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  author = {Simm, Gregor N. C. and Pinsler, Robert and {Hern{\'a}ndez-Lobato}, Jos{\'e} Miguel},
  editor = {III, Hal Daum{\'e} and Singh, Aarti},
  year = {2020},
  volume = {119},
  pages = {8959--8969},
  publisher = {{PMLR}},
  series = {Proceedings of Machine Learning Research}
  url = {http://proceedings.mlr.press/v119/simm20b.html}
}

@inproceedings{Simm2021SymmetryAware,
  title = {Symmetry-Aware Actor-Critic for 3D Molecular Design},
  author = {Gregor N. C. Simm and Robert Pinsler and G{\'a}bor Cs{\'a}nyi and Jos{\'e} Miguel Hern{\'a}ndez-Lobato},
  booktitle = {International Conference on Learning Representations},
  year = {2021},
  url = {https://openreview.net/forum?id=jEYKjPE1xYN}
}
```