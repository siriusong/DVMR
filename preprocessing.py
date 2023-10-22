import numpy as np
import pandas as pd
import argparse
import os
import re
import pickle

from rdkit import Chem
from tqdm import tqdm
from util.datautil import get_reaction_core_atoms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    default='USPTO50K',
                    help='dataset: USPTO50K or USPTO-full')

args = parser.parse_args()


# Get the mapping numbers in a SMARTS.
def get_idx(smarts_item):
    item = re.findall('(?<=:)\d+', smarts_item)
    item = list(map(int, item))
    return item


#  Get the dict maps each atom index to the mapping number.
def get_atomidx2mapidx(mol):
    atomidx2mapidx = {}
    for atom in mol.GetAtoms():
        atomidx2mapidx[atom.GetIdx()] = atom.GetAtomMapNum()
    return atomidx2mapidx


#  Get the dict maps each mapping number to the atom index .
def get_mapidx2atomidx(mol):
    mapidx2atomidx = {}
    mapidx2atomidx[0] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            mapidx2atomidx[0].append(atom.GetIdx())
        else:
            mapidx2atomidx[atom.GetAtomMapNum()] = atom.GetIdx()
    return mapidx2atomidx


# Get the reactant atom index list in the order of product atom index.
def get_order(product_mol, patomidx2pmapidx, rmapidx2ratomidx):
    order = []
    for atom in product_mol.GetAtoms():
        order.append(rmapidx2ratomidx[patomidx2pmapidx[atom.GetIdx()]])
    return order


def get_onehot(item, item_list):
    return list(map(lambda s: item == s, item_list))


def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def get_symbol_onehot(symbol):
    symbol_list = [
        'C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br', 'Li', 'Na', 'K',
        'Mg', 'B', 'Sn', 'I', 'Se', 'unk'
    ]
    if symbol not in symbol_list:
        print(symbol)
        symbol = 'unk'
    return list(map(lambda s: symbol == s, symbol_list))


def get_atom_feature(atom):
    degree_onehot = get_onehot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    H_num_onehot = get_onehot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    formal_charge = get_onehot(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
    chiral_tag = get_onehot(int(atom.GetChiralTag()), [0, 1, 2, 3])
    hybridization = get_onehot(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ])
    symbol_onehot = get_symbol_onehot(atom.GetSymbol())
    # Atom mass scaled to about the same range as other features
    atom_feature = degree_onehot + H_num_onehot + formal_charge + chiral_tag + hybridization + [
        atom.GetIsAromatic()
    ] + [atom.GetMass() * 0.01] + symbol_onehot
    return atom_feature


def get_atom_features(mol):
    feats = []
    for atom in mol.GetAtoms():
        feats.append(get_atom_feature(atom))
    return np.array(feats, dtype=np.float32)


def get_int_bond_type(bond_type):
    '''
        bond_type: SINGLE 1, DOUBLE 2, TRIPLE 3, AROMATIC 4
    '''
    if int(bond_type) == 12:
        return 4
    else:
        return int(bond_type)


def get_bond_features(mol):
    atom_num = len(mol.GetAtoms())
    # Bond feature dim is 12
    adj_array = np.zeros((atom_num, atom_num, 12), dtype=int)
    for bond in mol.GetBonds():
        bond_feat = bond_features(bond)
        adj_array[bond.GetBeginAtomIdx()][bond.GetEndAtomIdx()] = bond_feat
        adj_array[bond.GetEndAtomIdx()][bond.GetBeginAtomIdx()] = bond_feat
    return adj_array.astype(np.bool)


def bond_features(bond):
    """
    Builds a feature vector for a bond.
    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    bt = bond.GetBondType()
    fbond = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        (bond.GetIsConjugated() if bt is not None else 0),
        (bond.IsInRing() if bt is not None else 0)
    ]
    fbond += get_onehot(int(bond.GetStereo()), list(range(6)))
    return fbond


# Convert smarts to smiles by remove mapping numbers
def smarts2smiles(smarts, canonical=True):
    t = re.sub(':\d*', '', smarts)
    mol = Chem.MolFromSmiles(t, sanitize=False)
    return Chem.MolToSmiles(mol, canonical=canonical)


def del_index(smarts):
    t = re.sub(':\d*', '', smarts)
    return t


def onehot_encoding(x, total):
    return np.eye(total)[x]


def collate(data):
    return map(list, zip(*data))


# Split product smarts according to the target adjacent matrix
def get_smarts_pieces(mol, src_adj, target_adj, reacts, add_bond=False):
    m, n = src_adj.shape
    emol = Chem.EditableMol(mol)
    for j in range(m):
        for k in range(j + 1, n):
            if target_adj[j][k] == src_adj[j][k]:
                continue
            if 0 == target_adj[j][k]:
                emol.RemoveBond(j, k)
            elif add_bond:
                emol.AddBond(j, k, Chem.rdchem.BondType.SINGLE)
    synthon_smiles = Chem.MolToSmiles(emol.GetMol(), isomericSmiles=True)
    synthons = synthon_smiles.split('.')
    # Find the reactant with maximum common atoms for each synthon
    syn_idx_list = [get_idx(synthon) for synthon in synthons]
    react_idx_list = [get_idx(react) for react in reacts]

    react_max_common_synthon_index = []
    for react_idx in react_idx_list:
        react_common_idx_cnt = []
        for syn_idx in syn_idx_list:
            common_idx = list(set(syn_idx) & set(react_idx))
            react_common_idx_cnt.append(len(common_idx))
        max_cnt = max(react_common_idx_cnt)
        react_max_common_index = react_common_idx_cnt.index(max_cnt)
        react_max_common_synthon_index.append(react_max_common_index)
    react_synthon_index = np.argsort(react_max_common_synthon_index).tolist()
    reacts = [reacts[k] for k in react_synthon_index]

    return ' . '.join(synthons), ' . '.join(reacts)


# Split product smarts into synthons, src seq is synthon and tgt seq is reaction data
def generate_opennmt_data(save_dir, set_name, data_files):
    assert set_name in ['train', 'test', 'valid']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    src_data = []
    tgt_data = []
    for idx, f in tqdm(enumerate(data_files)):
        with open(os.path.join(save_dir, f), 'rb') as f:
            rxn_data = pickle.load(f)

        reaction_cls = rxn_data['rxn_type']
        reactant_mol = rxn_data['reactant_mol']
        product_mol = rxn_data['product_mol']
        product_adj = rxn_data['product_adj']
        target_adj = rxn_data['target_adj']
        reactant = Chem.MolToSmiles(reactant_mol)
        product = Chem.MolToSmiles(product_mol)
        reactants = reactant.split('.')

        src_item, tgt_item = get_smarts_pieces(product_mol, product_adj,
                                               target_adj, reactants)
        src_data.append([idx, reaction_cls, product, src_item])
        tgt_data.append(tgt_item)

    print('size', len(src_data))

    # Write data with reaction index
    with open(
            os.path.join('opennmt_data', 'src-{}.txt'.format(set_name)), 'w') as f:
        for src in src_data:
            f.write('{} [RXN_{}] {} [PREDICT] {}\n'.format(
                src[0], src[1], src[2], src[3]))

    with open(
            os.path.join('opennmt_data', 'tgt-{}.txt'.format(set_name)), 'w') as f:
        for tgt in tgt_data:
            f.write('{}\n'.format(tgt))
atom_feats = dict()
bond_statics = dict()
bond_types = dict()
def add_atom_dict(key):
    if key in atom_feats.keys():
        return atom_feats[key]
    else:
        atom_feats[key] = len(atom_feats)
        return atom_feats[key]
def add_bond_dict(key):
    bondtype,begin,end = key
    key2 = (bondtype,end,begin)
    if key in bond_statics.keys() :
        bond_statics[key] += 1
        return
    if key2 in bond_statics.keys():
        bond_statics[key2] += 1
        return
    bond_statics[key] = 1
    return
def add_bond_types(key):
    if key in bond_types.keys():
        return bond_types[key]
    else:
        bond_types[key] = len(bond_types)+1
        return bond_types[key]

def statistics_bond(mol):
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        begin_atom_feats = tuple(get_atom_feature(begin_atom))
        end_atom_feats = tuple(get_atom_feature(end_atom))
        begin_key = add_atom_dict(begin_atom_feats)
        end_key = add_atom_dict(end_atom_feats)

        bond_double = bond.GetBondTypeAsDouble()
        key = add_bond_types(bond_double)
        add_bond_dict((key,begin_key,end_key))
def get_bond_types(mol):

    atom_num = len(mol.GetAtoms())
    adj_array = np.zeros((atom_num, atom_num, 1), dtype=float)

    for bond in mol.GetBonds():
        key = bond.GetBondTypeAsDouble()
        adj_array[bond.GetBeginAtomIdx()][bond.GetEndAtomIdx()] = bond_types[key]
        adj_array[bond.GetEndAtomIdx()][bond.GetBeginAtomIdx()] = bond_types[key]
    return adj_array

def preprocess(save_dir, reactants, products,smiles, reaction_types=None,dataset='test'):
    """
    preprocess reaction data to extract graph adjacency matrix and features
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for index in tqdm(range(len(reactants))):
        product = products[index]
        reactant = reactants[index]
        product_mol = Chem.MolFromSmiles(product)
        reactant_mol = Chem.MolFromSmiles(reactant)
        if dataset == 'train':
            statistics_bond(product_mol)
        
        product_adj = Chem.rdmolops.GetAdjacencyMatrix(product_mol)
        product_adj = product_adj + np.eye(product_adj.shape[0])
        product_adj = product_adj.astype(np.bool)
        reactant_adj = Chem.rdmolops.GetAdjacencyMatrix(reactant_mol)
        reactant_adj = reactant_adj + np.eye(reactant_adj.shape[0])
        reactant_adj = reactant_adj.astype(np.bool)

        patomidx2pmapidx = get_atomidx2mapidx(product_mol)
        rmapidx2ratomidx = get_mapidx2atomidx(reactant_mol)
        order = get_order(product_mol, patomidx2pmapidx, rmapidx2ratomidx)
        target_adj = reactant_adj[order][:, order]
        _,prodct_edit,bond_edits,atom_edits = get_reaction_core_atoms(smiles[index])
        atom_num = len(product_mol.GetAtoms())
        atom_labels = np.zeros(atom_num)
        bond_labels = np.zeros((atom_num, atom_num))
        for atom_label in atom_edits.items():
            id,label = atom_label
            atom_labels[id] = label
        for bond_label in bond_edits.items():
            ids,label = bond_label
            id1 , id2 =ids
            bond_labels[id1][id2] = label
            bond_labels[id2][id1] = label
        product_bond_features = get_bond_features(product_mol)
        product_atom_features = get_atom_features(product_mol)
        product_bond_types = get_bond_types(product_mol)

        rxn_data = {
            'rxn_type': reaction_types[index],
            'product_adj': product_adj,
            'product_mol': product_mol,
            'product_bond_features': product_bond_features,
            'product_atom_features': product_atom_features,
            'target_adj': target_adj,
            'atom_label':atom_labels,
            'bond_label':bond_labels,
            'product_bond_type': product_bond_types,

            #'reactant_adj': reactant_adj,
            #'reactant_in_product_order': full_order,
            'reactant_mol': reactant_mol,
        }
        with open(os.path.join(save_dir, 'rxn_data_{}.pkl'.format(index)),
                  'wb') as f:
            pickle.dump(rxn_data, f)
def save_bond_dict(train_size):
    with open(os.path.join('data', str(train_size)+'_bonds_feats.pkl'),
            'wb') as f:
        bond_data ={
            'atom_feats_dict':atom_feats,
            'bond_static_dict':bond_statics,
            'bond_types':bond_types,
        }
        pickle.dump(bond_data, f)
if __name__ == '__main__':
    print('preprocessing dataset {}...'.format(args.dataset))
    assert args.dataset in ['USPTO50K', 'USPTO-full']

    datadir = 'data/{}/canonicalized_csv'.format(args.dataset)
    savedir = 'data/{}/'.format(args.dataset)

    for data_set in [ 'train','test', 'valid']:
        save_dir = os.path.join(savedir, data_set)
        csv_path = os.path.join(datadir, data_set + '.csv')
        csv = pd.read_csv(csv_path)
        reaction_list = csv['reactants>reagents>production']
        reactant_smarts_list = list(
            map(lambda x: x.split('>>')[0], reaction_list))
        product_smarts_list = list(
            map(lambda x: x.split('>>')[1], reaction_list))
        reaction_class_list = list(map(lambda x: int(x) - 1, csv['class']))
        # Extract product adjacency matrix and atom features
        preprocess(
            save_dir,
            reactant_smarts_list,
            product_smarts_list,
            reaction_list,
            reaction_class_list,
            dataset=data_set
        )