import torch
import warnings
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import numpy as np
from ..contracts import ISA_BATCH_CONTRACT, LEGACY_ISA_BATCH_CONTRACT


def _copy_float_target(target):
    """Return an independent float32 target tensor without tensor-copy warnings."""

    return torch.as_tensor(target, dtype=torch.float32).detach().clone()

class ISAGraphDataset_legacy(Dataset):
    batch_contract = LEGACY_ISA_BATCH_CONTRACT
    def __init__(self, graphs=None, target=None, smiles=None):
        if graphs is None: return
        self.graphs = graphs
        self.r_node = [g['r_nd'].x for g in graphs]
        self.r2r_edge = [g['r_nd', 'r2r', 'r_nd'].edge_attr for g in graphs]
        self.i_node = [g['i_nd'].x for g in graphs]
        self.d_node = [g['d_nd'].x for g in graphs]
        self.d2d_edge = [g['d_nd', 'd2d', 'd_nd'].edge_attr for g in graphs]
        self.target = _copy_float_target(target)
        self.smiles = smiles


    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.r_node[idx], self.r2r_edge[idx], self.i_node[idx], self.d_node[idx], self.d2d_edge[idx], self.target[idx], self.smiles[idx]
    
    def reload(self, data):
        self.graphs, self.r_node, self.r2r_edge, self.i_node, self.d_node, self.d2d_edge, self.target, self.smiles = data
        self.target = torch.stack(self.target)
        if self.target.dim() == 1:
            self.target = self.target.unsqueeze(-1)

    # def subDataset(self, idx):
    #     self.graphs = [self.graphs[i] for i in idx]
    #     self.r_node = [self.r_node[i] for i in idx]
    #     self.r2r_edge = [self.r2r_edge[i] for i in idx]
    #     self.i_node = [self.i_node[i] for i in idx]
    #     self.d_node = [self.d_node[i] for i in idx]
    #     self.d2d_edge = [self.d2d_edge[i] for i in idx]
    #     self.target = self.target[np.array(idx, dtype=int)]
    #     self.smiles = [self.smiles[i] for i in idx]

    def get_subDataset(self, idx):
        graphs = [self.graphs[i] for i in idx]
        r_node = [self.r_node[i] for i in idx]
        r2r_edge = [self.r2r_edge[i] for i in idx]
        i_node = [self.i_node[i] for i in idx]
        d_node = [self.d_node[i] for i in idx]
        d2d_edge = [self.d2d_edge[i] for i in idx]
        target = [self.target[i] for i in idx]
        smiles = [self.smiles[i] for i in idx]
        
        dataset = ISAGraphDataset_legacy()
        dataset.reload((graphs, r_node, r2r_edge, i_node, d_node, d2d_edge, target, smiles))
        return dataset

    @staticmethod
    def collate(samples):
        graphs, r_node, r2r_edge, i_node, d_node, d2d_edge, target, smiles = map(list, zip(*samples))
        batched_graph = Batch.from_data_list(graphs)
        return batched_graph, torch.concat(r_node,dim=0), torch.concat(r2r_edge,dim=0), torch.concat(i_node,dim=0), torch.concat(d_node,dim=0), torch.concat(d2d_edge,dim=0), torch.stack(target,dim=0), smiles
    
    @staticmethod
    def unwrapper(graph, r_node, r2r_edge, i_node, d_node, d2d_edge, target, smiles, device='cpu'):
        graph = graph.to(device=device)
        r_node = r_node.float().to(device=device)
        r2r_edge = r2r_edge.float().to(device=device)
        i_node = i_node.float().to(device=device)
        d_node = d_node.float().to(device=device)
        d2d_edge = d2d_edge.float().to(device=device)
        target = target.float().to(device=device)
        return {'graph':graph, 'r_node':r_node, 'r_edge':r2r_edge, 'i_node':i_node, 'd_node':d_node, 'd_edge':d2d_edge, 'target':target, 'smiles':smiles}
    
    
class ISAGraphDataset(ISAGraphDataset_legacy):
    batch_contract = ISA_BATCH_CONTRACT
    def __init__(self, graphs=None, target=None, smiles=None, numeric_inputs=None, row_indices=None):
        if graphs is None: 
            self.graphs = {}
            self.target = None
            self.smiles = {}
            self.data_keys = []
            return


        for key in graphs:
            setattr(self, key + '_graphs', graphs[key])
            try:
                setattr(self, key + '_r_node', [g['r_nd'].x for g in graphs[key]])
            except KeyError:
                warnings.warn(
                    f"Graph key {key!r} has no 'r_nd' node type; using zero features.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                setattr(self, key + '_r_node', [torch.zeros((g['r_nd'].num_nodes, 1)) for g in graphs[key]])

            try:
                setattr(self, key + '_r2r_edge', [g['r_nd', 'r2r', 'r_nd'].edge_attr for g in graphs[key]])
            except KeyError:
                warnings.warn(
                    f"Graph key {key!r} has no 'r2r' edge type; using zero features.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                setattr(self, key + '_r2r_edge', [torch.zeros((g['r_nd', 'r2r', 'r_nd'].num_edges, 1)) for g in graphs[key]])
            # if 'i_nd' in graphs[key][0].nodes:
            #     setattr(self, key + '_i_node', [g.nodes['i_nd'].data['f'] for g in graphs[key]])
            # else:
            #     setattr(self, key + '_i_node', [torch.zeros((g.num_nodes(), 0)) for g in graphs[key]])
            try:
                setattr(self, key + '_i2i_edge', [g['i_nd', 'i2i', 'i_nd'].edge_attr for g in graphs[key]])
            except KeyError:
                warnings.warn(
                    f"Graph key {key!r} has no 'i2i' edge type; using zero features.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                setattr(self, key + '_i2i_edge', [torch.zeros((g['i_nd', 'i2i', 'i_nd'].num_edges, 1)) for g in graphs[key]])



            # if 'i2i' in graphs[key][0].nodes:
            #     setattr(self, key + '_i_node', [g.nodes['i_nd'].data['f'] for g in graphs[key]])
            # else:
            #     setattr(self, key + '_i_node', [torch.zeros((g.num_nodes(), 0)) for g in graphs[key]])
            try:
                setattr(self, key + '_i_node', [g['i_nd'].x for g in graphs[key]])
            except KeyError:
                warnings.warn(
                    f"Graph key {key!r} has no 'i_nd' node type; using zero features.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                setattr(self, key + '_i_node', [torch.zeros((g['i_nd'].num_nodes, 1)) for g in graphs[key]])

            try:
                setattr(self, key + '_d_node', [g['d_nd'].x for g in graphs[key]])
            except KeyError:
                warnings.warn(
                    f"Graph key {key!r} has no 'd_nd' node type; using zero features.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                setattr(self, key + '_d_node', [torch.zeros((g['d_nd'].num_nodes, 1)) for g in graphs[key]])

            try:
                setattr(self, key + '_d2d_edge', [g['d_nd', 'd2d', 'd_nd'].edge_attr for g in graphs[key]])
            except KeyError:
                warnings.warn(
                    f"Graph key {key!r} has no 'd2d' edge type; using zero features.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                setattr(self, key + '_d2d_edge', [torch.zeros((g['d_nd', 'd2d', 'd_nd'].num_edges, 1)) for g in graphs[key]])

            # if 'd_nd' in graphs[key][0].nodes:
            #     setattr(self, key + '_d_node', [g.nodes['d_nd'].data['f'] for g in graphs[key]])
            # else:
            #     setattr(self, key + '_d_node', [torch.zeros((g.num_nodes(), 0)) for g in graphs[key]])

            # if 'd2d' in graphs[key][0].edges:
            #     setattr(self, key + '_d2d_edge', [g.edges['d2d'].data['dist'] for g in graphs[key]])
            # else:
            #     setattr(self, key + '_d2d_edge', [torch.zeros((g.num_edges(), 0)) for g in graphs[key]])

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
                if hasattr(self, key + '_smiles'):
                    item[key + '_smiles'] = getattr(self, key + '_smiles')[idx]
                # if hasattr(self, key + '_node_feature'):
                #     nf = getattr(self, key + '_node_feature')[idx]
                #     if type(nf) is not torch.Tensor:
                #         nf = torch.tensor(nf, dtype=torch.float32)
                #     item[key + '_node_feature'] = nf
                # if hasattr(self, key + '_edge_feature'):
                #     ef = getattr(self, key + '_edge_feature')[idx]
                #     if type(ef) is not torch.Tensor:
                #         ef = torch.tensor(ef, dtype=torch.float32)
                #     item[key + '_edge_feature'] = ef
                if hasattr(self, key + '_r_node'):
                    f = getattr(self, key + '_r_node')[idx]
                    if type(f) is not torch.Tensor:
                        f = torch.tensor(f, dtype=torch.float32)
                    item[key + '_r_node'] = f
                if hasattr(self, key + '_r2r_edge'):
                    f = getattr(self, key + '_r2r_edge')[idx]
                    if type(f) is not torch.Tensor:
                        f = torch.tensor(f, dtype=torch.float32)
                    item[key + '_r2r_edge'] = f
                if hasattr(self, key + '_i_node'):
                    f = getattr(self, key + '_i_node')[idx]
                    if type(f) is not torch.Tensor:
                        f = torch.tensor(f, dtype=torch.float32)
                    item[key + '_i_node'] = f
                if hasattr(self, key + '_i2i_edge'):
                    f = getattr(self, key + '_i2i_edge')[idx]
                    if type(f) is not torch.Tensor:
                        f = torch.tensor(f, dtype=torch.float32)
                    item[key + '_i2i_edge'] = f
                if hasattr(self, key + '_d_node'):
                    f = getattr(self, key + '_d_node')[idx]
                    if type(f) is not torch.Tensor:
                        f = torch.tensor(f, dtype=torch.float32)
                    item[key + '_d_node'] = f
                if hasattr(self, key + '_d2d_edge'):
                    f = getattr(self, key + '_d2d_edge')[idx]
                    if type(f) is not torch.Tensor:
                        f = torch.tensor(f, dtype=torch.float32)
                    item[key + '_d2d_edge'] = f

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
                # setattr(self, key + '_node_feature', [getattr(self, key + '_node_feature')[i] for i in idx])
                # setattr(self, key + '_edge_feature', [getattr(self, key + '_edge_feature')[i] for i in idx])
            if hasattr(self, key + '_r_node'):
                setattr(self, key + '_r_node', [getattr(self, key + '_r_node')[i] for i in idx])
            if hasattr(self, key + '_r2r_edge'):
                setattr(self, key + '_r2r_edge', [getattr(self, key + '_r2r_edge')[i] for i in idx])
            if hasattr(self, key + '_i_node'):
                setattr(self, key + '_i_node', [getattr(self, key + '_i_node')[i] for i in idx])
            if hasattr(self, key + '_i2i_edge'):
                setattr(self, key + '_i2i_edge', [getattr(self, key + '_i2i_edge')[i] for i in idx])
            if hasattr(self, key + '_d_node'):
                setattr(self, key + '_d_node', [getattr(self, key + '_d_node')[i] for i in idx])
            if hasattr(self, key + '_d2d_edge'):
                setattr(self, key + '_d2d_edge', [getattr(self, key + '_d2d_edge')[i] for i in idx])

            if hasattr(self, key + '_var'):
                var = getattr(self, key + '_var')
                if torch.is_tensor(var):
                    setattr(self, key + '_var', var[torch.tensor(idx, dtype=torch.long)])
                else:
                    setattr(self, key + '_var', var[np.array(idx, dtype=int)])
            if hasattr(self, key + '_smiles'):
                setattr(self, key + '_smiles', [getattr(self, key + '_smiles')[i] for i in idx])
        if self.target is not None:
            self.target = self.target[np.array(idx, dtype=int)]
        if self.original_row_index is not None:
            self.original_row_index = self.original_row_index[torch.as_tensor(idx, dtype=torch.long)]

    def get_subDataset(self, idx):
        dataset = ISAGraphDataset()
        dataset.reload([self[i] for i in idx])
        return dataset


    @staticmethod
    def collate(samples):
        batched_data = {}
        for key in samples[0].keys():
            if key.endswith('_graphs'):
                batched_data[key] = Batch.from_data_list([s[key] for s in samples])
            # elif key.endswith('_node_feature'):
            #     batched_data[key] = torch.concat([s[key] for s in samples], dim=0)
            # elif key.endswith('_edge_feature'):
            #     batched_data[key] = torch.concat([s[key] for s in samples], dim=0)
            elif key.endswith('_r_node') or key.endswith('_i_node') or key.endswith('_d_node'):
                batched_data[key] = torch.concat([s[key] for s in samples], dim=0)
            elif key.endswith('_r2r_edge') or key.endswith('_i2i_edge') or key.endswith('_d2d_edge'):
                batched_data[key] = torch.concat([s[key] for s in samples], dim=0)
            elif key.endswith('_var'):
                batched_data[key] = torch.stack([s[key].reshape(-1) for s in samples], dim=0)
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
            elif key.endswith('_r_node') or key.endswith('_i_node') or key.endswith('_d_node'):
                batched_data[key] = batched_data[key].float().to(device=device)
            elif key.endswith('_r2r_edge') or key.endswith('_i2i_edge') or key.endswith('_d2d_edge'):
                batched_data[key] = batched_data[key].float().to(device=device)
            elif key.endswith('_var'):
                batched_data[key] = batched_data[key].float().to(device=device)
            elif key == 'target':
                batched_data[key] = batched_data[key].float().to(device=device)
            elif key == 'original_row_index':
                batched_data[key] = batched_data[key].long().to(device=device)
        return batched_data
