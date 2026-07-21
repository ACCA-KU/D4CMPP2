import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import numpy as np
from ..contracts import (
    GENERAL_BATCH_CONTRACT,
    LEGACY_GENERAL_BATCH_CONTRACT,
    LEGACY_SOLVENT_BATCH_CONTRACT,
)


def _copy_float_target(target):
    """Return an independent float32 target tensor without tensor-copy warnings."""

    return torch.as_tensor(target, dtype=torch.float32).detach().clone()

class GraphDataset_legacy(Dataset):
    batch_contract = LEGACY_GENERAL_BATCH_CONTRACT
    def __init__(self, graphs=None, target=None, smiles=None):
        if graphs is None: return
        self.graphs = graphs
        self.node_feature = [g.x for g in graphs]
        self.edge_feature = [g.edge_attr for g in graphs]
        self.target = _copy_float_target(target)
        self.smiles = smiles

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.node_feature[idx], self.edge_feature[idx], self.target[idx], self.smiles[idx]
    
    # reload the dataset with new data
    def reload(self, data):
        self.graphs, self.node_feature, self.edge_feature, self.target, self.smiles = data

        self.target= torch.stack(self.target,dim=0)
        if self.target.dim() == 1:
            self.target = self.target.unsqueeze(-1)

    # get a subset of the dataset with given indices
    def get_subDataset(self, idx):
        graphs = [self.graphs[i] for i in idx]
        node_feature = [self.node_feature[i] for i in idx]
        edge_feature = [self.edge_feature[i] for i in idx]
        target = [self.target[i] for i in idx]
        smiles = [self.smiles[i] for i in idx]
        
        dataset = GraphDataset_legacy()
        dataset.reload((graphs, node_feature, edge_feature, target, smiles))
        return dataset
        

    @staticmethod
    def collate(samples):
        graphs, node_feature, edge_feature, target, smiles = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, torch.concat(node_feature,dim=0), torch.concat(edge_feature,dim=0), torch.stack(target,dim=0), smiles
    
    @staticmethod
    def unwrapper(batch_graph, node_feature, edge_feature, target, smiles,device='cpu'):
        batch_graph = batch_graph.to(device=device)
        node_feature = node_feature.float().to(device=device)
        edge_feature = edge_feature.float().to(device=device)
        target = target.float().to(device=device)
        return {"graph":batch_graph, "node_feats":node_feature, "edge_feats":edge_feature, "target":target, "smiles":smiles}

class GraphDataset_withSolv(GraphDataset_legacy):
    batch_contract = LEGACY_SOLVENT_BATCH_CONTRACT
    def __init__(self, graphs=None, solv_graphs=None, target=None, smiles=None, solv_smiles=None):
        super().__init__(graphs, target, smiles)
        if graphs is None: return
        self.solv_graphs = solv_graphs
        self.solv_node_feature = [g.x for g in solv_graphs]
        self.solv_edge_feature = [g.edge_attr for g in solv_graphs]
        self.solv_smiles = solv_smiles

    def __getitem__(self, idx):
        args = super().__getitem__(idx)
        return args + (self.solv_graphs[idx], self.solv_node_feature[idx], self.solv_edge_feature[idx], self.solv_smiles[idx])
    
    def reload(self, data):
        self.graphs, self.node_feature, self.edge_feature, self.target, self.smiles, self.solv_graphs, self.solv_node_feature, self.solv_edge_feature, self.solv_smiles = data
        
        self.target= torch.stack(self.target,dim=0)
        if self.target.dim() == 1:
            self.target = self.target.unsqueeze(-1)


    # def subDataset(self, idx):
    #     super().subDataset(idx)
    #     self.solv_graphs = [self.solv_graphs[i] for i in idx]
    #     self.solv_node_feature = [self.solv_node_feature[i] for i in idx]
    #     self.solv_edge_feature = [self.solv_edge_feature[i] for i in idx]
    #     self.solv_smiles = [self.solv_smiles[i] for i in idx]

    def get_subDataset(self, idx):
        graphs = [self.graphs[i] for i in idx]
        node_feature = [self.node_feature[i] for i in idx]
        edge_feature = [self.edge_feature[i] for i in idx]
        target = [self.target[i] for i in idx]
        smiles = [self.smiles[i] for i in idx]        
        
        solv_graphs = [self.solv_graphs[i] for i in idx]
        solv_node_feature = [self.solv_node_feature[i] for i in idx]
        solv_edge_feature = [self.solv_edge_feature[i] for i in idx]
        solv_smiles = [self.solv_smiles[i] for i in idx]

        dataset = GraphDataset_withSolv()
        dataset.reload((graphs, node_feature, edge_feature, target, smiles, solv_graphs, solv_node_feature, solv_edge_feature, solv_smiles))
        return dataset


    @staticmethod
    def collate(samples):
        graphs, node_feature, edge_feature, target, smiles, solv_graphs, solv_node_feature, solv_edge_feature, solv_smiles = map(list, zip(*samples))

        batched_graph = Batch.from_data_list(graphs)
        batched_solv_graph = Batch.from_data_list(solv_graphs)
        return batched_graph, torch.concat(node_feature,dim=0), torch.concat(edge_feature,dim=0), batched_solv_graph, torch.concat(solv_node_feature,dim=0), torch.concat(solv_edge_feature,dim=0), torch.stack(target,dim=0), smiles, solv_smiles
    
    @staticmethod
    def unwrapper(batch_graph, node_feature, edge_feature, batch_solv_graph, solv_node_feature, solv_edge_feature, target, smiles, solv_smiles, device='cpu'):
        batch_graph = batch_graph.to(device=device)
        node_feature = node_feature.float().to(device=device)
        edge_feature = edge_feature.float().to(device=device)
        batch_solv_graph = batch_solv_graph.to(device=device)
        solv_node_feature = solv_node_feature.float().to(device=device)
        solv_edge_feature = solv_edge_feature.float().to(device=device)
        target = target.float().to(device=device)

        return {
            # Canonical keys consumed by MolecularNetwork.input_contract.
            "compound_graphs": batch_graph,
            "compound_node_feature": node_feature,
            "compound_edge_feature": edge_feature,
            "compound_smiles": smiles,
            "solvent_graphs": batch_solv_graph,
            "solvent_node_feature": solv_node_feature,
            "solvent_edge_feature": solv_edge_feature,
            "solvent_smiles": solv_smiles,
            # Legacy aliases retained for saved/custom network compatibility.
            "graph": batch_graph,
            "node_feats": node_feature,
            "edge_feats": edge_feature,
            "solv_graph": batch_solv_graph,
            "solv_node_feats": solv_node_feature,
            "solv_edge_feats": solv_edge_feature,
            "target": target,
            "smiles": smiles,
            "solv_smiles": solv_smiles,
        }
    

class GraphDataset(GraphDataset_legacy):
    batch_contract = GENERAL_BATCH_CONTRACT
    def __init__(self, graphs : dict = None, numeric_inputs : dict = None, target : list = None, smiles : dict = None, row_indices=None):
        if graphs is None: 
            self.graphs = {}
            self.target = None
            self.smiles = {}
            self.data_keys = []
            return

        for key in graphs:
            setattr(self, key + '_graphs', graphs[key])
            setattr(self, key + '_node_feature', [g.x for g in graphs[key]])
            setattr(self, key + '_edge_feature', [g.edge_attr for g in graphs[key]])

        if numeric_inputs is not None:
            for key in numeric_inputs:
                setattr(self, key + '_var', numeric_inputs[key])
        else:
            numeric_inputs = {}

        for key in smiles:
            if key not in graphs:
                raise ValueError(f"Key '{key}' in smiles is not found in graphs.")
            setattr(self, key + '_smiles', smiles[key])

        self.target = _copy_float_target(target) if target is not None else None
        self.original_row_index = (
            torch.as_tensor(row_indices, dtype=torch.long) if row_indices is not None else None
        )
        self.data_keys = list(graphs.keys()) + list(numeric_inputs.keys()) + ['target']


    def __len__(self):
        if getattr(self, 'target', None) is not None:
            return len(self.target)
        else:
            if len(self.data_keys) == 0:
                return 0
            key = self.data_keys[0]
            if hasattr(self, key + '_graphs'):
                return len(getattr(self, key + '_graphs'))
            elif hasattr(self, key + '_var'):
                return len(getattr(self, key + '_var'))
            elif hasattr(self, key + '_smiles'):
                return len(getattr(self, key + '_smiles'))
            else:
                return 0


    def __getitem__(self, idx):
        item = {}
        for key in self.data_keys:
            if hasattr(self, key + '_graphs'):
                item[key + '_graphs'] = getattr(self, key + '_graphs')[idx]
                if hasattr(self, key + '_node_feature'):
                    nf = getattr(self, key + '_node_feature')[idx]
                    if type(nf) is not torch.Tensor:
                        nf = torch.tensor(nf, dtype=torch.float32)
                    item[key + '_node_feature'] = nf
                if hasattr(self, key + '_edge_feature'):
                    ef = getattr(self, key + '_edge_feature')[idx]
                    if type(ef) is not torch.Tensor:
                        ef = torch.tensor(ef, dtype=torch.float32)
                    item[key + '_edge_feature'] = ef
                if hasattr(self, key + '_smiles'):
                    item[key + '_smiles'] = getattr(self, key + '_smiles')[idx]
            elif hasattr(self, key + '_var'):
                var = getattr(self, key + '_var')[idx]
                if torch.is_tensor(var):
                    var = var.float()
                else:
                    var = torch.tensor(var, dtype=torch.float32)
                item[key + '_var'] = var
            elif hasattr(self, key + '_smiles'):
                item[key + '_smiles'] = getattr(self, key + '_smiles')[idx]
        if hasattr(self, 'target') and self.target is not None:
            item['target'] = self.target[idx]
        if self.original_row_index is not None:
            item['original_row_index'] = self.original_row_index[idx]
        return item

    def reload(self, data):
        if len(data) == 0:
            self.target = None
            self.data_keys = []
            return

        new_data_keys = []
        for key in data[0]:
            values = [d[key] for d in data]
            if key == 'target':
                if len(values) == 0:
                    setattr(self, key, None)
                elif torch.is_tensor(values[0]):
                    target = torch.stack(values, dim=0).float()
                    if target.dim() == 1:
                        target = target.unsqueeze(-1)
                    setattr(self, key, target)
                else:
                    target = torch.tensor(values, dtype=torch.float32)
                    if target.dim() == 1:
                        target = target.unsqueeze(-1)
                    setattr(self, key, target)
            elif key == 'original_row_index':
                setattr(self, key, torch.as_tensor(values, dtype=torch.long))
            elif key.endswith('_var'):
                if len(values) == 0:
                    setattr(self, key, torch.empty((0,), dtype=torch.float32))
                elif torch.is_tensor(values[0]):
                    setattr(self, key, torch.stack(values, dim=0).float())
                else:
                    setattr(self, key, torch.tensor(values, dtype=torch.float32))
            else:
                setattr(self, key, values)
            if key.endswith('_graphs'):
                new_data_keys.append(key[:-7])  # Remove '_graphs'
            elif key.endswith('_var'):
                new_data_keys.append(key[:-4])  # Remove '_var'
            elif key == 'target':
                new_data_keys.append(key)
        self.data_keys = new_data_keys


    def subDataset(self, idx):
        for key in self.data_keys:
            if hasattr(self, key + '_graphs'):
                setattr(self, key + '_graphs', [getattr(self, key + '_graphs')[i] for i in idx])
                setattr(self, key + '_node_feature', [getattr(self, key + '_node_feature')[i] for i in idx])
                setattr(self, key + '_edge_feature', [getattr(self, key + '_edge_feature')[i] for i in idx])
            elif hasattr(self, key + '_var'):   
                var = getattr(self, key + '_var')
                if torch.is_tensor(var):
                    setattr(self, key + '_var', var[torch.tensor(idx, dtype=torch.long)])
                else:
                    setattr(self, key + '_var', var[np.array(idx, dtype=int)])
            elif hasattr(self, key + '_smiles'):
                setattr(self, key + '_smiles', [getattr(self, key + '_smiles')[i] for i in idx])
        if self.target is not None:
            self.target = self.target[np.array(idx, dtype=int)]
        if self.original_row_index is not None:
            self.original_row_index = self.original_row_index[torch.as_tensor(idx, dtype=torch.long)]

    def get_subDataset(self, idx):
        dataset = GraphDataset()
        dataset.reload([self[i] for i in idx])
        return dataset


    @staticmethod
    def collate(samples):
        batched_data = {}
        for key in samples[0].keys():
            if key.endswith('_graphs'):
                batched_data[key] = Batch.from_data_list([s[key] for s in samples])
            elif key.endswith('_node_feature'):
                batched_data[key] = torch.concat([s[key] for s in samples], dim=0)
            elif key.endswith('_edge_feature'):
                batched_data[key] = torch.concat([s[key] for s in samples], dim=0)
            elif key.endswith('_var'):
                batched_data[key] = torch.stack([s[key] for s in samples], dim=0)
            elif key.endswith('_smiles'):
                batched_data[key] = [s[key] for s in samples]
            elif key == 'target':
                batched_data[key] = torch.stack([s[key].reshape(-1) for s in samples], dim=0)
            elif key == 'original_row_index':
                batched_data[key] = torch.stack([s[key].reshape(()) for s in samples], dim=0)
        return batched_data

    @staticmethod
    def unwrapper(device='cpu', **batched_data):
        for key in batched_data.keys():
            if key.endswith('_graphs'):
                batched_data[key] = batched_data[key].to(device=device)
            elif key.endswith('_node_feature') or key.endswith('_edge_feature'):
                batched_data[key] = batched_data[key].float().to(device=device)
            elif key.endswith('_var'):
                batched_data[key] = batched_data[key].float().to(device=device)
            elif key == 'target':
                batched_data[key] = batched_data[key].float().to(device=device)
            elif key == 'original_row_index':
                batched_data[key] = batched_data[key].long().to(device=device)
        return batched_data

    
        
