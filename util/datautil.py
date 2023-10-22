import re
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm

def parse_reaction_roles(rxn_smiles, as_what="smiles"):
    """ Convert a reaction SMILES string to lists of reactants, reagents and products in various data formats. """

    # Split the reaction SMILES string by the '>' symbol to obtain the reactants and products.
    rxn_roles = rxn_smiles.split(">")

    # NOTE: In some cases, there can be additional characters on the product side separated by a whitespace. For this
    # reason the product side string is always additionally split by the whitespace and the only the first element is
    # considered.

    # Parse the original SMILES strings from the reaction string including the reaction atom mappings.
    if as_what == "smiles":
        return [x for x in rxn_roles[0].split(".") if x != ""],\
               [x for x in rxn_roles[1].split(".") if x != ""],\
               [x for x in rxn_roles[2].split(" ")[0].split(".") if x != ""]

    # Parse the original SMILES strings from the reaction string excluding the reaction atom mappings.
    elif as_what == "smiles_no_maps":
        return [re.sub(r":[-+]?[0-9]+", "", x) for x in rxn_roles[0].split(".") if x != ""],\
               [re.sub(r":[-+]?[0-9]+", "", x) for x in rxn_roles[1].split(".") if x != ""],\
               [re.sub(r":[-+]?[0-9]+", "", x) for x in rxn_roles[2].split(" ")[0].split(".") if x != ""]

    # Parse the lists of atom map numbers from the reactions SMILES string.
    elif as_what == "atom_maps":
        return [[int(x[1:]) for x in re.findall(r":[-+]?[0-9]+", y)] for y in rxn_roles[0].split(".") if y != ""],\
               [[int(x[1:]) for x in re.findall(r":[-+]?[0-9]+", y)] for y in rxn_roles[1].split(".") if y != ""],\
               [[int(x[1:]) for x in re.findall(r":[-+]?[0-9]+", y)]
                for y in rxn_roles[2].split(" ")[0].split(".") if y != ""]

    # Parse the lists of number of mapped atoms per reactant and per product. Reagents usually do not contain any
    # mapping, but they are included here for the sake of consistency.
    elif as_what == "mapping_numbers":
        return [len([el for el in rxn_roles[0].split(".") if el != ""]),
                len([el for el in rxn_roles[1].split(".") if el != ""]),
                len([el for el in rxn_roles[2].split(" ")[0].split(".") if el != ""])]

    # Parsings that include initial conversion to RDKit Mol objects and need to include sanitization: mol, mol_no_maps,
    # canonical_smiles and canonical_smiles_no_maps.
    elif as_what in ["mol", "mol_no_maps", "canonical_smiles", "canonical_smiles_no_maps"]:
        reactants, reagents, products = [], [], []

        # Iterate through all of the reactants.
        for reactant in rxn_roles[0].split("."):
            if reactant != "" and as_what in ["mol", "canonical_smiles"]:
                mol_maps = AllChem.MolFromSmiles(reactant)
                AllChem.SanitizeMol(mol_maps)

                if as_what == "mol":
                    reactants.append(mol_maps)
                    continue

                if as_what == "canonical_smiles":
                    reactants.append(AllChem.MolToSmiles(mol_maps, canonical=True))
                    continue

            elif reactant != "" and as_what in ["mol_no_maps", "canonical_smiles_no_maps"]:
                mol_no_maps = AllChem.MolFromSmiles(re.sub(r":[-+]?[0-9]+", "", reactant))
                AllChem.SanitizeMol(mol_no_maps)

                if as_what == "mol_no_maps":
                    reactants.append(mol_no_maps)
                    continue

                if as_what == "canonical_smiles_no_maps":
                    reactants.append(AllChem.MolToSmiles(mol_no_maps, canonical=True))
                    continue

        # Iterate through all of the reagents.
        for reagent in rxn_roles[1].split("."):
            if reagent != "" and as_what in ["mol", "canonical_smiles"]:
                mol_maps = AllChem.MolFromSmiles(reagent)
                AllChem.SanitizeMol(mol_maps)

                if as_what == "mol":
                    reagents.append(mol_maps)
                    continue

                if as_what == "canonical_smiles":
                    reagents.append(AllChem.MolToSmiles(mol_maps, canonical=True))
                    continue

            elif reagent != "" and as_what in ["mol_no_maps", "canonical_smiles_no_maps"]:
                mol_no_maps = AllChem.MolFromSmiles(re.sub(r":[-+]?[0-9]+", "", reagent))
                AllChem.SanitizeMol(mol_no_maps)

                if as_what == "mol_no_maps":
                    reagents.append(mol_no_maps)
                    continue

                if as_what == "canonical_smiles_no_maps":
                    reagents.append(AllChem.MolToSmiles(mol_no_maps, canonical=True))
                    continue

        # Iterate through all of the reactants.
        for product in rxn_roles[2].split(" ")[0].split("."):
            if product != "" and as_what in ["mol", "canonical_smiles"]:
                mol_maps = AllChem.MolFromSmiles(product)
                AllChem.SanitizeMol(mol_maps)

                if as_what == "mol":
                    products.append(mol_maps)
                    continue

                if as_what == "canonical_smiles":
                    products.append(AllChem.MolToSmiles(mol_maps, canonical=True))
                    continue

            elif product != "" and as_what in ["mol_no_maps", "canonical_smiles_no_maps"]:
                mol_no_maps = AllChem.MolFromSmiles(re.sub(r":[-+]?[0-9]+", "", product))
                AllChem.SanitizeMol(mol_no_maps)

                if as_what == "mol_no_maps":
                    products.append(mol_no_maps)
                    continue

                if as_what == "canonical_smiles_no_maps":
                    products.append(AllChem.MolToSmiles(mol_no_maps, canonical=True))
                    continue

        return reactants, reagents, products

    # Raise exception for any other keyword.
    else:
        raise Exception("Unknown parsing type. Select one of the following: "
                        "'smiles', 'smiles_no_maps', 'atom_maps', 'mapping_numbers', 'mol', 'mol_no_maps', "
                        "'canonical_smiles', 'canonical_smiles_no_maps'.")
def molecule_is_mapped(mol):
    """ Checks if a molecule created from a RDKit Mol object or a SMILES string contains at least one mapped atom."""

    # If it is a RDKit Mol object, check if any atom map number has a value other than zero.
    if not isinstance(mol, str):
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() != 0:
                return True
        return False

    # If it is a SMILES string, check if the string contains the symbol ":" used for mapping.
    else:
        return ":" in mol
def same_neighbourhood_size(atom_index_1, molecule_1, atom_index_2, molecule_2):
    """ Checks whether the same atoms in two different molecules (e.g., reactant and product molecules) have the same
        neighbourhood size. """

    if len(molecule_1.GetAtomWithIdx(atom_index_1).GetNeighbors()) != \
            len(molecule_2.GetAtomWithIdx(atom_index_2).GetNeighbors()):
        return False
    return True


def same_neighbour_atoms(atom_index_1, molecule_1, atom_index_2, molecule_2):
    """ Checks whether the same atoms in two different molecules (e.g., reactant and product molecules) have retained
        the same types of chemical elements in their immediate neighbourhood according to reaction mapping numbers. """

    neighbourhood_1, neighbourhood_2 = [], []

    for i in molecule_1.GetAtomWithIdx(atom_index_1).GetNeighbors():
        neighbourhood_1.append((i.GetAtomMapNum(), i.GetSymbol(), i.GetFormalCharge(),
                                i.GetNumRadicalElectrons(), i.GetTotalValence()))
    for j in molecule_2.GetAtomWithIdx(atom_index_2).GetNeighbors():
        neighbourhood_2.append((j.GetAtomMapNum(), j.GetSymbol(), j.GetFormalCharge(),
                                j.GetNumRadicalElectrons(), j.GetTotalValence()))

    return sorted(neighbourhood_1) == sorted(neighbourhood_2)


def same_neighbour_bonds(atom_index_1, molecule_1, atom_index_2, molecule_2):
    """ Checks whether the same atoms in two different molecules (e.g., reactant and product molecules) have retained
        the same types of chemical bonds amongst each other in their immediate neighbourhood. """

    neighbourhood_1, neighbourhood_2 = [], []

    for i in molecule_1.GetAtomWithIdx(atom_index_1).GetNeighbors():
        neighbourhood_1.append((i.GetAtomMapNum(),
                                str(molecule_1.GetBondBetweenAtoms(atom_index_1, i.GetIdx()).GetBondType())))
    for j in molecule_2.GetAtomWithIdx(atom_index_2).GetNeighbors():
        neighbourhood_2.append((j.GetAtomMapNum(),
                                str(molecule_2.GetBondBetweenAtoms(atom_index_2, j.GetIdx()).GetBondType())))
    #b=molecule_1.GetBondBetweenAtoms(atom_index_1,i.GetIdx()).GetIdx()
    return sorted(neighbourhood_1) == sorted(neighbourhood_2)


def get_difference_bond(atom_index_1, molecule_1, atom_index_2, molecule_2):
    #molecule_1 pro molecule_2 rec
    neighbourhood_1_bondtype, neighbourhood_2_bondtype = [], []
    neighbourhood_1_atomtype, neighbourhood_2_atomtype = [], []
    molmaptoid = {}
    a=molecule_1.GetAtomWithIdx(atom_index_1).GetAtomMapNum()
    for i in molecule_1.GetAtomWithIdx(atom_index_1).GetNeighbors():
        neighbourhood_1_bondtype.append((i.GetAtomMapNum(),
                                str(molecule_1.GetBondBetweenAtoms(atom_index_1, i.GetIdx()).GetBondType())))
        neighbourhood_1_atomtype.append((i.GetAtomMapNum(), i.GetSymbol(), i.GetFormalCharge(),
                                i.GetNumRadicalElectrons(), i.GetTotalValence()))
        molmaptoid[i.GetAtomMapNum()] = i.GetIdx()
    for j in molecule_2.GetAtomWithIdx(atom_index_2).GetNeighbors():
        neighbourhood_2_bondtype.append((j.GetAtomMapNum(),
                                str(molecule_2.GetBondBetweenAtoms(atom_index_2, j.GetIdx()).GetBondType())))
        neighbourhood_2_atomtype.append((j.GetAtomMapNum(), j.GetSymbol(), j.GetFormalCharge(),
                                j.GetNumRadicalElectrons(), j.GetTotalValence()))
    id_list = []
    bond_results = {}
    atom_results = {}
    atom_results[atom_index_1] = 1
    for id, pro in enumerate(neighbourhood_1_bondtype):
        if pro not in  neighbourhood_2_bondtype:
            id_list.append(id)
    if len(id_list) >0:
        for id in id_list:
            atom_mapnum , _ = neighbourhood_1_bondtype[id]
            atomid = molmaptoid[atom_mapnum]
            #bond_results[molecule_1.GetBondBetweenAtoms(atom_index_1,atomid).GetIdx()] = 1
            bond_results[(atom_index_1,atomid)] = 1
    id_list = []
    for id, pro in enumerate(neighbourhood_1_atomtype):
        if pro not in neighbourhood_2_atomtype:
            id_list.append(id)
    if len(id_list) >0:
        for id in id_list:
            atom_mapnum , _,_,_,_= neighbourhood_1_atomtype[id]
            atomid = molmaptoid[atom_mapnum]
            if (atom_index_1,atomid) not in bond_results.keys():
                bond_results[(atom_index_1,atomid)] = 2
    return bond_results, atom_results

def get_reaction_core_atoms(rsmiles):
    reactants, _, products = parse_reaction_roles(rsmiles, as_what="mol")
    reactants_final = [set() for _ in range(len(reactants))]
    products_final = [set() for _ in range(len(products))]
    bond_edits = {}
    atom_edits = {}
    count = 0
    for p_ind, product in enumerate(products):

        for r_ind, reactant in enumerate(reactants):
            for p_atom in product.GetAtoms():
                if p_atom.GetAtomMapNum() <= 0:
                    products_final[p_ind].add(p_atom.GetIdx())
                    continue
                for r_atom in reactant.GetAtoms():
                    max_amap = max([atom.GetAtomMapNum() for atom in reactant.GetAtoms()])

                    if molecule_is_mapped(reactant) and r_atom.GetAtomMapNum() <= 0:
                        reactants_final[r_ind].add(r_atom.GetIdx())
                        r_atom.SetAtomMapNum(max_amap + 1) #对recant中的没有mapnum的原子分配
                        max_amap += 1
                        continue
                    if p_atom.GetAtomMapNum() == r_atom.GetAtomMapNum():
                        if not same_neighbourhood_size(p_atom.GetIdx(), product, r_atom.GetIdx(), reactant) or \
                                not same_neighbour_atoms(p_atom.GetIdx(), product, r_atom.GetIdx(), reactant) or \
                                not same_neighbour_bonds(p_atom.GetIdx(), product, r_atom.GetIdx(), reactant):
                            reactants_final[r_ind].add(r_atom.GetIdx())
                            products_final[p_ind].add(p_atom.GetIdx())
                            bond_edit,atom_edit =get_difference_bond(p_atom.GetIdx(),product,r_atom.GetIdx(),reactant)
                            bond_edits.update(bond_edit)
                            atom_edits.update(atom_edit)
    return reactants_final, products_final,bond_edits,atom_edits

if __name__ == '__main__':
    data_dir = '../../datasets/uspto-50k/canonicalized_train.csv'
    pf=pd.read_csv(data_dir)
    smiles = pf["reactants>reagents>production"]
    for smile in tqdm(smiles):
        get_reaction_core_atoms(smile)