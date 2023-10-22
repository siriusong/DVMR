import torch
from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import networkx as nx
import dgl
from rdkit import  Chem
from collections import Counter
from tqdm import tqdm

class RetroCenterDatasets(Dataset):
    def __init__(self, root, data_split):
        self.root = root
        self.data_split = data_split

        self.data_dir = os.path.join(root, self.data_split)
        self.data_files = [
            f for f in os.listdir(self.data_dir) if f.endswith('.pkl')
        ]
        self.data_files.sort()

        self.disconnection_num = []



        count = 0
        cnt = Counter()
        for data_file in self.data_files:
            with open(os.path.join(self.data_dir, data_file), 'rb') as f:
                reaction_data = pickle.load(f)
            xa = reaction_data['product_adj']
            ya = reaction_data['target_adj']
            res = xa & (ya == False)
            res = np.sum(np.sum(res)) // 2
            cnt[res] += 1
            if res >= 2:
                res = 2
            count += 1

            self.disconnection_num.append(res)
        # print(cnt)
    def get_bond_feats(self):
        single_feats = []
        double_feats = []
        AROMATIC_feats = []
        TRIPLE_feats = []
        for key in self.bond_static_dict.keys():
            bond_label,atom_id1,atom_id2 = key
            atom_id1_feats = np.array(self.rever_atom_feats_dict[atom_id1]).astype(np.float32)
            atom_id2_feats = np.array(self.rever_atom_feats_dict[atom_id2]).astype(np.float32)
            feats_1 = np.concatenate((atom_id1_feats,atom_id2_feats))
            feats_2 = np.concatenate((atom_id2_feats,atom_id1_feats))
            if bond_label == 1:
                single_feats.append(feats_1)
                single_feats.append(feats_2)
            elif bond_label == 1.5:
                AROMATIC_feats.append(feats_1)
                AROMATIC_feats.append(feats_2)
            elif bond_label == 2:
                double_feats.append(feats_1)
                double_feats.append(feats_2)
            elif bond_label == 3:
                TRIPLE_feats.append(feats_1)
                TRIPLE_feats.append(feats_2)
            else:
                print('error {}'.format(str(bond_label)))
                return
        return torch.tensor(single_feats),torch.tensor(double_feats),torch.tensor(AROMATIC_feats),torch.tensor(TRIPLE_feats)
    def get_bond_feats2(self,train_size = 0):
        if train_size > 0 :
            with open(os.path.join(self.root, str(train_size)+'_bonds_feats.pkl'), 'rb') as f:
                bonds_feats = pickle.load(f)
            atom_feats_dict = bonds_feats['atom_feats_dict']
            self.rever_atom_feats_dict = {}
            self.bond_static_dict = bonds_feats['bond_static_dict']
            self.bond_types = bonds_feats['bond_types']
        else :
            with open(os.path.join(self.root, 'bonds_feats2.pkl'), 'rb') as f:
                bonds_feats = pickle.load(f)
            atom_feats_dict = bonds_feats['atom_feats_dict']
            self.rever_atom_feats_dict = {}
            self.bond_static_dict = bonds_feats['bond_static_dict']
            self.bond_types = bonds_feats['bond_types']
        for key,value in atom_feats_dict.items():
            self.rever_atom_feats_dict[value] = key
        bond_list = [[] for _ in range(len(self.bond_types)+1)]
        for key in self.bond_static_dict.keys():
            bond_label,atom_id1,atom_id2 = key
            atom_id1_feats = np.array(self.rever_atom_feats_dict[atom_id1]).astype(np.float32)
            atom_id2_feats = np.array(self.rever_atom_feats_dict[atom_id2]).astype(np.float32)
            feats_1 = np.concatenate((atom_id1_feats,atom_id2_feats))
            feats_2 = np.concatenate((atom_id2_feats,atom_id1_feats))
            bond_list[bond_label].append(feats_1)
            bond_list[bond_label].append(feats_2)

        return   list(map(lambda x: torch.tensor(x).float(), bond_list))

    def get_bond_feats3(self):
        begin_list = [[] for _ in range(len(self.bond_types)+1)]
        end_list = [[] for _ in range(len(self.bond_types)+1)]
        for key in self.bond_static_dict.keys():
            bond_label,atom_id1,atom_id2 = key
            atom_id1_feats = np.array(self.rever_atom_feats_dict[atom_id1]).astype(np.float32)
            atom_id2_feats = np.array(self.rever_atom_feats_dict[atom_id2]).astype(np.float32)
            # feats_1 = np.concatenate((atom_id1_feats,atom_id2_feats))
            # feats_2 = np.concatenate((atom_id2_feats,atom_id1_feats))
            begin_list[bond_label].append(atom_id1_feats)
            begin_list[bond_label].append(atom_id2_feats)

            end_list[bond_label].append(atom_id2_feats)
            end_list[bond_label].append(atom_id1_feats)

        return   (list(map(lambda x: torch.tensor(x).float(), begin_list)),list(map(lambda x: torch.tensor(x).float(), end_list)))
    def __getitem__(self, index):
        # single,double,AROMATIC,TRIPLE = self.get_bond_feats()
        # new_features = []
        # zeros = np.zeros_like(single)
        # newsingle = single.detach().cpu().numpy()
        # newdouble = double.detach().cpu().numpy()
        # newaromatic = AROMATIC.detach().cpu().numpy()
        # newtriple = TRIPLE.detach().cpu().numpy()
        with open(os.path.join(self.data_dir, self.data_files[index]),
                  'rb') as f:
            reaction_data = pickle.load(f)

        x_atom = reaction_data['product_atom_features'].astype(np.float32)
        x_pattern_feat = reaction_data['pattern_feat'].astype(np.float32)
        # x_pattern_feat = np.random.random((x_atom.shape[0],657))
        x_bond = reaction_data['product_bond_features'].astype(np.float32)
        x_adj = reaction_data['product_adj']
        y_adj = reaction_data['target_adj']
        product_smiles = Chem.MolToSmiles(reaction_data['product_mol'])
        rxn_class = reaction_data['rxn_type']
        rxn_class = np.eye(10)[rxn_class]
        product_atom_num = len(x_atom)
        # bond_label = reaction_data['bond_label']
        # atom_label = reaction_data['atom_label']
        rxn_class = np.expand_dims(rxn_class, 0).repeat(product_atom_num,
                                                        axis=0)
        bond_type = reaction_data['product_bond_type']
        # x_groups=reaction_data['Group']
        # Construct graph and add edge data
        bond_type = torch.from_numpy(bond_type[x_adj])
        graph = nx.from_numpy_matrix(x_adj)
       # data存类型 作为索引 重排特征
        x_graph = dgl.from_networkx(graph)
        node_num=x_graph.num_nodes()
        # print(index)
        x_mol = reaction_data['product_mol']
        smiles = Chem.MolToSmiles(x_mol)
        atoms = x_mol.GetAtoms()
        atoms_num = len(atoms)
        if node_num != atoms_num or node_num != x_atom.shape[0]:
            print(os.path.join(self.data_dir, self.data_files[index]))
        w=torch.from_numpy(x_bond[x_adj])
        x_graph.edata['w']=w#edate['w']
        x_graph.edata['type'] = bond_type
        bond = x_graph.edata['w']
        # feat = x_graph.edata['type'].cpu().numpy()
        # b = feat[:,0]
        # for i in b:
        #     if i == 0:
        #         new_features.append(zeros)
        #     elif i == 1:
        #         new_features.append(newsingle)
        #     elif i == 2:
        #         new_features.append()
        disconnection_num = self.disconnection_num[index]

        return rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, y_adj,disconnection_num

    def __len__(self):
        return len(self.data_files)


if __name__ == '__main__':
    # savedir = 'data/USPTO50K/'
    # for data_set in ['train']:
    #     save_dir = os.path.join(savedir, data_set)
    #     train_data = RetroCenterDatasets(root=savedir, data_split=data_set)
    #     train_data.__getitem__(1)
    #     print(train_data.data_files[:100])
    dir = 'data/USPTO50K/'
    for data_set in ['train','valid','test']:
        traindata = RetroCenterDatasets(root=dir,data_split=data_set)
        for i in tqdm(range(len(traindata))):
            rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, y_adj,disconnection_num=traindata[0]