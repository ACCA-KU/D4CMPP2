"""Positive-only ISA Analyzer implementations."""

from __future__ import annotations

import io
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import rdkit.Chem as Chem
from rdkit.Chem.Draw import rdMolDraw2D

from .MolAnalyzer import MolAnalyzer, MolAnalyzer_v2
from .interpretation import ISAInterpreter


class ISAAnalyzer(MolAnalyzer):
    """Analyzer for ISA models that expose one positive attribution stream."""

    def __init__(self, model_path, save_result=False, **kwargs):
        super().__init__(model_path, save_result, **kwargs)
        self.data_keys = ["prediction", "positive", "fragments"]
        self.for_pickle = ["fragments"]
        if getattr(self.tm, "get_score", None) is None:
            raise ValueError("This model does not support ISAAnalyzer.")
        self.is_score_by_group = True
        self._interpreter = ISAInterpreter(
            self._core,
            score_keys=("positive",),
        )

    def analyze_rows(self, *args, include_features=False, **kwargs):
        """Return row-aligned positive ISA scores."""

        return self._interpreter.analyze(
            *args,
            include_features=include_features,
            **kwargs,
        )

    def plot_analysis(self, analysis_row, *, score="positive", **kwargs):
        """Plot one already-calculated immutable ISA analysis row."""

        if score not in analysis_row.scores:
            raise ValueError(
                f"Score {score!r} is unavailable. Available scores: "
                f"{list(analysis_row.scores)}."
            )
        smiles = analysis_row.inputs[self.molecule_columns[0]]
        return self._plot_score(
            smiles,
            {"positive": analysis_row.atom_scores(score)},
            **kwargs,
        )

    def check_score_by_group(self):
        if getattr(self.dm.gg, "sculptor", None) is None:
            self.is_score_by_group = True
            return
        smiles = "CC(C)(C)OC(=O)C1=CC=CC=C1C(=O)O"
        loader, _ = self.prepare_temp_data([smiles])
        output = self.tm.get_score(self.nm, loader)
        if "positive" not in output:
            raise ValueError("The ISA model did not return a positive score.")
        score = output["positive"].detach().cpu().numpy()
        fragment_count = len(self.get_fragment(smiles))
        atom_count = Chem.MolFromSmiles(smiles).GetNumAtoms()
        if score.shape[0] == fragment_count:
            self.is_score_by_group = True
        elif score.shape[0] == atom_count:
            self.is_score_by_group = False
        else:
            raise ValueError(
                "The positive score length matches neither the fragment count "
                f"({fragment_count}) nor atom count ({atom_count}); got {score.shape[0]}."
            )

    def get_score(self, smiles):
        """Return the positive attribution score for one molecule."""

        cached = self.load_data(smiles, "positive")
        if cached is not None:
            return {"positive": cached}
        loader, _ = self.prepare_temp_data([smiles])
        output = self.tm.get_score(self.nm, loader)
        score = output["positive"].detach().cpu().numpy()
        result = {"positive": self.get_group_score(smiles, score)}
        self.save_data(smiles, result)
        return result

    def get_fragment(self, smiles, get_index=False):
        """Return saved-model fragments for one molecule."""

        sculptor = self.dm.gg.sculptor
        fragments = sculptor.fragmentation_with_condition(
            Chem.MolFromSmiles(smiles),
            draw=False,
            get_index=False,
        )
        self.save_data(smiles, {"fragments": fragments})
        if get_index:
            return [fragment.atoms for fragment in fragments]
        return fragments

    def get_scores(self, smiles_list):
        """Return cached or calculated positive scores keyed by SMILES."""

        results = {}
        pending = []
        for smiles in smiles_list:
            cached = self.load_data(smiles, "positive")
            if cached is None:
                pending.append(smiles)
            else:
                results[smiles] = {"positive": cached}
        if pending:
            valid_smiles, calculated = self._get_all_scores(pending)
            results.update(zip(valid_smiles, calculated))
        return results

    def _get_all_scores(self, smiles_list):
        loader, valid = self.prepare_temp_data(smiles_list)
        if len(valid) == 0:
            return [], []
        output = self.tm.get_score(self.nm, loader)
        scores = output["positive"].detach().cpu().numpy()
        predictions = output.get("prediction")
        if predictions is not None:
            predictions = predictions.detach().cpu().numpy()
        results = []
        offset = 0
        for index, smiles in enumerate(valid["compound"]):
            fragments = self.get_fragment(smiles, get_index=True)
            score = scores[offset:offset + len(fragments)]
            offset += len(fragments)
            result = {
                "positive": self.get_group_score(smiles, score, fragments),
            }
            if predictions is not None:
                result["prediction"] = predictions[index]
            self.save_data(smiles, result)
            results.append(result)
        return valid["compound"], results

    def plot_score(self, smiles, **kwargs):
        """Plot the positive attribution score for one molecule."""

        return self._plot_score(smiles, self.get_score(smiles), **kwargs)

    def _plot_score(
        self,
        smiles,
        score,
        *,
        atom_with_index=False,
        score_scaler=lambda value: value,
        ticks=(0, 0.25, 0.5, 0.75, 1),
        tick_type="manual",
        rot=0,
        line_width=2.0,
        figsize=1,
        with_colorbar=True,
        **kwargs,
    ):
        molecule = Chem.MolFromSmiles(smiles)
        raw_score = np.asarray(score["positive"])
        atom_score = self.get_atom_score(
            smiles,
            np.asarray(score_scaler(raw_score)),
        )
        color_map = LinearSegmentedColormap.from_list(
            "isa_score",
            [(0, "orangered"), (0.5, "white"), (1, "royalblue")],
        )
        png = showAtomHighlight(
            molecule,
            atom_score,
            color_map,
            atom_with_index,
            rot,
            line_width,
            figsize=figsize,
        )
        image = Image.open(io.BytesIO(png))
        figure, axis = plt.subplots(1, 1, figsize=(16 * figsize, 11 * figsize))
        axis.imshow(image)
        axis.axis("off")
        if with_colorbar:
            colorbar = plt.colorbar(
                plt.cm.ScalarMappable(cmap=color_map),
                ax=axis,
                shrink=0.7,
            )
            colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
            labels = ticks
            if tick_type == "minmax":
                labels = np.round(
                    np.linspace(np.min(raw_score), np.max(raw_score), 5),
                    3,
                )
            colorbar.set_ticklabels(labels)
        plt.show()
        return {"positive": atom_score}

    def get_subgroup_score_bin(self, smiles_list):
        """Group positive scores by fragment SMILES."""

        fragment_scores = {}
        for smiles, result in self.get_scores(smiles_list).items():
            values = np.asarray(result["positive"]).reshape(-1, 1)
            for index, fragment in enumerate(self.get_fragment(smiles)):
                current = values[index]
                if fragment.smiles in fragment_scores:
                    current = np.concatenate(
                        [fragment_scores[fragment.smiles], current],
                        axis=0,
                    )
                fragment_scores[fragment.smiles] = current
        return fragment_scores

    def plot_score_hist(self, smiles_list, xlim=(0, 1), bins=40):
        """Plot and return all positive scores."""

        values = []
        for result in self.get_scores(smiles_list).values():
            values.extend(set(np.asarray(result["positive"]).flatten()))
        plt.hist(
            values,
            bins=np.arange(xlim[0], xlim[1], 1.0 / bins),
            label="positive",
        )
        plt.xlim(*xlim)
        plt.show()
        return values

    def plot_subgroup_score_histogram(
        self,
        smiles_list,
        nums=10,
        bins=40,
        xlim=(0, 1),
    ):
        """Plot the most frequent fragment-level positive scores."""

        fragments = self.get_subgroup_score_bin(smiles_list)
        fragments = dict(
            sorted(fragments.items(), key=lambda item: len(item[1]), reverse=True)[
                :nums
            ]
        )
        for fragment, values in fragments.items():
            plt.hist(
                values,
                bins=np.arange(xlim[0], xlim[1], 1.0 / bins),
                alpha=0.5,
                label=fragment,
            )
        plt.xlim(*xlim)
        plt.legend()
        plt.title("Positive")
        plt.show()
        return fragments

    def plot_subgroup_score_histogram_byone(
        self,
        smiles_list,
        nums=10,
        bins=40,
        xlim=(0, 1),
        target_subgroups=None,
    ):
        """Plot each selected fragment's positive-score histogram."""

        fragments = self.get_subgroup_score_bin(smiles_list)
        fragments = dict(
            sorted(fragments.items(), key=lambda item: len(item[1]), reverse=True)[
                :nums
            ]
        )
        if target_subgroups:
            fragments = {
                key: value
                for key, value in fragments.items()
                if key in target_subgroups
            }
        figure, axes = plt.subplots(
            max(len(fragments), 1),
            1,
            figsize=(5, 2 * max(len(fragments), 1)),
            squeeze=False,
        )
        for axis, (fragment, values) in zip(axes[:, 0], fragments.items()):
            axis.hist(
                values,
                bins=np.arange(xlim[0], xlim[1], 1.0 / bins),
                label="positive",
            )
            axis.set_ylabel(fragment)
            axis.set_xlim(*xlim)
            axis.legend()
        plt.show()
        return fragments

    def get_atom_score(self, smiles, score, frag=None):
        molecule = Chem.MolFromSmiles(smiles)
        if len(score) < molecule.GetNumAtoms():
            score = self.group_score2atom_score(smiles, score, frag)
        if len(score) != molecule.GetNumAtoms():
            raise ValueError(
                "The length of score and the number of atoms do not match."
            )
        return score

    def get_group_score(self, smiles, score, frag=None):
        if getattr(self, "is_score_by_group", None) is None:
            return score
        if frag is None:
            frag = self.get_fragment(smiles, get_index=True)
        if len(frag) < score.shape[0]:
            score = self.atom_score2group_score(smiles, score, frag)
        if len(frag) != score.shape[0]:
            raise ValueError(
                "The number of fragments and the length of score do not match."
            )
        return score

    def group_score2atom_score(self, smiles, group_score, frag=None):
        molecule = Chem.MolFromSmiles(smiles)
        if frag is None:
            frag = self.get_fragment(smiles, get_index=True)
        if len(frag) != group_score.shape[0]:
            raise ValueError(
                "The number of fragments and the length of group_score do not match."
            )
        atom_score = np.zeros(
            (molecule.GetNumAtoms(),) + np.asarray(group_score).shape[1:],
            dtype=np.asarray(group_score).dtype,
        )
        for atoms, value in zip(frag, group_score):
            atom_score[list(atoms)] = value
        return atom_score

    def atom_score2group_score(self, smiles, atom_score, frag=None):
        if frag is None:
            frag = self.get_fragment(smiles, get_index=True)
        if len(atom_score) != sum(len(atoms) for atoms in frag):
            raise ValueError(
                "The length of atom_score and the number of atoms do not match."
            )
        return np.asarray([atom_score[atoms[0]] for atoms in frag])


class ISAAnalyzer_v2(MolAnalyzer_v2, ISAAnalyzer):
    """Version 2 positive-only ISA Analyzer."""

    def __init__(self, model_path, save_result=False, **kwargs):
        super().__init__(model_path, save_result, **kwargs)
        if getattr(self.tm, "get_score", None) is None:
            raise ValueError("This model does not support ISAAnalyzer.")

    def check_score_by_group(self):
        """Score alignment is validated by :class:`ISAInterpreter`."""

    def get_score(self, *args, **kwargs):
        analysis = self.analyze_rows(*args, include_features=False, **kwargs)
        values = [dict(row.scores) for row in analysis]
        return values[0] if len(values) == 1 else values

    def plot_score(
        self,
        *args,
        draw_mol_columns=None,
        feature=None,
        **kwargs,
    ):
        if draw_mol_columns is None:
            raise ValueError("draw_mol_columns must be specified.")
        if draw_mol_columns not in self.molecule_columns:
            raise ValueError(
                f"{draw_mol_columns} is not a valid molecule column. "
                f"Choose from {self.molecule_columns}."
            )
        if feature is None:
            raise ValueError("feature must be specified.")
        inputs, plot_kwargs = self.handle_positional_args(args, kwargs)
        smiles = inputs.get(draw_mol_columns, [None])[0]
        if smiles is None:
            raise ValueError(f"{draw_mol_columns} must be specified in kwargs.")
        score = {"positive": self.get_score(*args, **kwargs)[feature]}
        return self._plot_score(smiles, score, **plot_kwargs)


class ISAAnalyzer_v1p3(ISAAnalyzer_v2):
    """Deprecated compatibility name for :class:`ISAAnalyzer_v2`."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ISAAnalyzer_v1p3 is deprecated; use ISAAnalyzer_v2 or "
            "Analyzer(...) instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def showAtomHighlight(
    mol,
    score,
    color_map,
    atom_with_index=True,
    rot=0,
    line_width=1.0,
    figsize=1,
):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if atom_with_index:
        mol = mol_with_atom_index(mol)
    highlighted_atoms = list(range(mol.GetNumAtoms()))
    atom_colors = {
        index: color_map(np.asarray(score[index]).item())
        for index in highlighted_atoms
    }
    highlighted_bonds = []
    bond_colors = {}
    for index in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(index)
        source = bond.GetBeginAtomIdx()
        target = bond.GetEndAtomIdx()
        if np.array_equal(score[source], score[target]):
            highlighted_bonds.append(index)
            bond_colors[index] = color_map(np.asarray(score[source]).item())
    drawer = rdMolDraw2D.MolDraw2DCairo(
        int(1000 * figsize),
        int(800 * figsize),
    )
    options = drawer.drawOptions()
    options.rotate = rot
    options.bondLineWidth = line_width
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlighted_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=highlighted_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mol_with_atom_index(mol):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    for index in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(index).SetProp("molAtomMapNumber", str(index))
    return mol
