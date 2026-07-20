from rdkit import Chem
import torch
import numpy as np
import traceback
import warnings
from collections import deque
from torch_geometric.data import HeteroData

from .MolGraphGenerator import MolGraphGenerator
from D4CMPP2.src.utils.featureizer import InvalidAtomError
from D4CMPP2.src.utils.sculptor import SubgroupSplitter

class ISAGraphGenerator(MolGraphGenerator):
    def __init__(self, frag_ref=None, sculptor_index=(6,2,0)):
        self.sculptor = SubgroupSplitter(frag_ref,
                                        get_index=True,
                                        split_order=sculptor_index[0],
                                        combine_rest_order=sculptor_index[1],
                                        absorb_neighbor_order=sculptor_index[2],
                                        overlapped_ring_combine=True
                                )
        self.r_node_dim = None
        self.i_node_dim = None
        self.d_node_dim = None
        self.r_edge_dim = None
        self.i_edge_dim = None
        self.d_edge_dim = None
        self.node_dim= None
        self.edge_dim = None
        super().__init__()


        self.set_feature_dim()
        self.node_dim = self.r_node_dim 
        self.edge_dim = self.r_edge_dim
        self.verbose = True

    def set_feature_dim(self):
        try:
            get_graph = self.get_graph('FC1CCCCC1CCCOCCC')
        except InvalidAtomError as e:
            if self.verbose:
                warnings.warn(
                    f"ISA feature-dimension probe encountered an invalid atom: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def get_graph(self, smi, **kwargs):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: 
            raise Exception("Invalid SMILES: failed to generate mol object")
        if kwargs.get("explicit_h",False):
            mol = Chem.AddHs(mol)
        g = self.generate_graph(mol, **kwargs)
        atom_feature = self.af(mol)
        bond_feature = self.bf(mol)

        g['r_nd'].x = torch.tensor(atom_feature).float()
        if self.r_node_dim is None:
            self.r_node_dim = atom_feature.shape[1]

        relation = ('r_nd', 'r2r', 'r_nd')
        if mol.GetNumBonds() == 0:
            edge_dim = self.r_edge_dim
            if edge_dim is None:
                edge_dim = self.bf(Chem.MolFromSmiles("C-C")).shape[1]
            edge_count = g[relation].edge_index.shape[1]
            edata = torch.zeros((edge_count, edge_dim), dtype=torch.float32)
        else:
            edata = torch.tensor(bond_feature).float()
            edata = torch.cat([edata,edata],dim=0)
        g[relation].edge_attr = edata
        if self.r_edge_dim is None:
            self.r_edge_dim = edata.shape[1]

        g['i_nd'].x = torch.zeros((g['i_nd'].num_nodes, 1)).float()
        if self.i_node_dim is None:
            self.i_node_dim = 1
        if self.i_edge_dim is None:
            self.i_edge_dim = 0

        g['d_nd'].x = torch.zeros((g['d_nd'].num_nodes, 1)).float()
        if self.d_node_dim is None:
            self.d_node_dim = 1
        if self.d_edge_dim is None:
            self.d_edge_dim = 0


        return g
    
    def generate_sub_graph(self, mol, frags):
        src, dst = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            for frag in frags:
                if start in frag:
                    if end in frag:
                        src.append(start)
                        dst.append(end)
                        break
                    else:
                        break
                elif end in frag:
                    if start in frag:
                        src.append(start)
                        dst.append(end)
                        break
                    else:
                        break
                    
        return (src+dst, dst+src)
        

    def generate_dot_graph(self, mol, frags, max_dist = 4):
        src, dst = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            for i,frag in enumerate(frags):
                if start in frag:
                    start_frag = i
                if end in frag:
                    end_frag = i
            if start_frag != end_frag:
                src.append(start_frag)
                dst.append(end_frag)
        if len(src) == 0:
            return ([],[]), []
        adjacency = [[] for _ in frags]
        for start, end in zip(src, dst):
            adjacency[start].append(end)
            adjacency[end].append(start)

        out_src, out_dst, distances = [], [], []
        for root in range(len(frags)):
            shortest = [-1] * len(frags)
            shortest[root] = 0
            queue = deque([root])
            while queue:
                current = queue.popleft()
                if shortest[current] >= max_dist:
                    continue
                for neighbor in adjacency[current]:
                    if shortest[neighbor] == -1:
                        shortest[neighbor] = shortest[current] + 1
                        queue.append(neighbor)
            for target in range(root + 1, len(frags)):
                distance = shortest[target]
                if 0 < distance <= max_dist:
                    out_src.append(root)
                    out_dst.append(target)
                    distances.append(distance)

        directed_distances = distances + distances
        features = np.zeros((len(directed_distances), max_dist), dtype=np.float32)
        for row, distance in enumerate(directed_distances):
            features[row, distance - 1] = 1.0
        return (out_src + out_dst, out_dst + out_src), features

    def generate_graph(self,mol, **kwargs):
        max_dist = kwargs.get('max_dist',4)
        num_atoms = mol.GetNumAtoms()
        mol_data = self.generate_mol_graph(mol)
        
        frag = self.sculptor.fragmentation_with_condition(mol)
        frag_data = self.generate_sub_graph(mol, frag) 

        i2d_src = []
        i2d_dst = []
        for i, f in enumerate(frag):
            for a in f:
                i2d_src.append(a)
                i2d_dst.append(i)

        if len(frag) ==1:
            dot_data = ([0],[0])
            dist = np.zeros((1,max_dist))
        else:
            dot_data, dist = self.generate_dot_graph(mol, frag, max_dist = max_dist)

        graph_data = {
            ('r_nd', 'r2r', 'r_nd'): mol_data,
            ('r_nd', 'r2i', 'i_nd'): (list(range(num_atoms)), list(range(num_atoms))),
            ('i_nd', 'i2i', 'i_nd'): frag_data,
            ('i_nd', 'i2d', 'd_nd'): (i2d_src, i2d_dst),
            ('d_nd', 'd2d', 'd_nd'): dot_data,
            ('d_nd', 'd2r', 'r_nd'): (i2d_dst, i2d_src),

        }


        g = HeteroData()
        g['r_nd'].num_nodes = num_atoms
        g['i_nd'].num_nodes = num_atoms
        g['d_nd'].num_nodes = len(frag)
        for edge_type, (src, dst) in graph_data.items():
            g[edge_type].edge_index = torch.tensor([src, dst], dtype=torch.long)
        g['r_nd', 'r2r', 'r_nd'].edge_attr = torch.empty((len(mol_data[0]), 0))
        g['i_nd', 'i2i', 'i_nd'].edge_attr = torch.empty((len(frag_data[0]), 0))
        g['d_nd', 'd2d', 'd_nd'].edge_attr = torch.tensor(dist).float()
        if self.d_edge_dim is None:
            self.d_edge_dim = dist.shape[1]

        return g

    def get_empty_graph(self):
        graph = HeteroData()
        for node_type, dim in (("r_nd", self.r_node_dim), ("i_nd", 1), ("d_nd", 1)):
            graph[node_type].x = torch.zeros((0, dim or 1), dtype=torch.float32)
        relations = (
            ("r_nd", "r2r", "r_nd"),
            ("r_nd", "r2i", "i_nd"),
            ("i_nd", "i2i", "i_nd"),
            ("i_nd", "i2d", "d_nd"),
            ("d_nd", "d2d", "d_nd"),
            ("d_nd", "d2r", "r_nd"),
        )
        for relation in relations:
            graph[relation].edge_index = torch.empty((2, 0), dtype=torch.long)
        graph["r_nd", "r2r", "r_nd"].edge_attr = torch.zeros((0, self.r_edge_dim or 1))
        graph["i_nd", "i2i", "i_nd"].edge_attr = torch.zeros((0, 0))
        graph["d_nd", "d2d", "d_nd"].edge_attr = torch.zeros((0, self.d_edge_dim or 1))
        return graph
