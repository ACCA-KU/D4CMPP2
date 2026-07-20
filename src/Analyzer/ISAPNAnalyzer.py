"""Positive-negative ISA Analyzer implementations."""

from __future__ import annotations

import warnings

import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from PIL import Image
import rdkit.Chem as Chem

from .ISAAnalyzer import ISAAnalyzer, ISAAnalyzer_v2, showAtomHighlight
from .interpretation import ISAInterpreter


class _ISAPNAnalyzerMixin:
    """ISATPN-only positive/negative score and hidden-feature behavior."""

    def _configure_isapn_interpreter(self):
        self.data_keys = [
            "prediction",
            "positive",
            "negative",
            "feature_P",
            "feature_N",
            "fragments",
        ]
        self._interpreter = ISAInterpreter(
            self._core,
            score_keys=("positive", "negative"),
            feature_keys=("feature_P", "feature_N"),
            score_mode="fragment",
        )

    def analyze_rows(self, *args, include_features=True, **kwargs):
        """Return ISATPN positive/negative scores and optional P/N features."""

        return self._interpreter.analyze(
            *args, include_features=include_features, **kwargs
        )

    def get_score(self, *args, **kwargs):
        """Return row-aligned positive and negative ISATPN scores."""

        analysis = self.analyze_rows(*args, include_features=False, **kwargs)
        values = [dict(row.scores) for row in analysis]
        return values[0] if len(values) == 1 else values

    def get_scores(self, smiles_list):
        """Return positive and negative scores keyed by SMILES."""

        analysis = self.analyze_rows(smiles_list, include_features=False)
        molecule_column = self.molecule_columns[0]
        return {
            row.inputs[molecule_column]: dict(row.scores)
            for row in analysis
        }

    def plot_analysis(self, analysis_row, **kwargs):
        """Plot both score streams from one immutable ISATPN analysis row."""

        smiles = analysis_row.inputs[self.molecule_columns[0]]
        return self._plot_score(
            smiles,
            {
                "positive": analysis_row.atom_scores("positive"),
                "negative": analysis_row.atom_scores("negative"),
            },
            **kwargs,
        )

    def plot_score(self, *args, draw_mol_columns=None, **kwargs):
        """Calculate and plot both ISATPN score streams."""

        if draw_mol_columns is None:
            draw_mol_columns = self.molecule_columns[0]
        if draw_mol_columns not in self.molecule_columns:
            raise ValueError(
                f"{draw_mol_columns} is not a valid molecule column. "
                f"Choose from {self.molecule_columns}."
            )
        inputs, plot_kwargs = self.handle_positional_args(args, kwargs)
        smiles = inputs.get(draw_mol_columns, [None])[0]
        if smiles is None:
            raise ValueError(f"{draw_mol_columns} must be specified.")
        scores = self.get_score(**inputs)
        return self._plot_score(smiles, scores, **plot_kwargs)

    def get_feature(self, *args, feature=None, **kwargs):
        """Return row-aligned ISATPN positive/negative hidden features."""

        analysis = self.analyze_rows(*args, include_features=True, **kwargs)
        rows = []
        for row in analysis:
            values = dict(row.features)
            if feature is not None:
                if feature not in values:
                    raise ValueError(
                        f"Feature {feature!r} is unavailable. "
                        f"Available features: {list(values)}."
                    )
                rows.append(values[feature])
            else:
                rows.append(values)
        return rows[0] if len(rows) == 1 else rows

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
        locate="right",
        figsize=1,
        only_total=False,
        with_colorbar=True,
        **kwargs,
    ):
        """Plot ISATPN positive, negative, and combined attribution scores."""

        molecule = Chem.MolFromSmiles(smiles)
        positive = self.get_atom_score(
            smiles,
            np.asarray(score_scaler(score["positive"])),
        )
        negative = self.get_atom_score(
            smiles,
            np.asarray(score_scaler(score["negative"])),
        )
        raw_total = (1 + positive - negative) / 2
        total = np.asarray(score_scaler(raw_total))
        color_map = LinearSegmentedColormap.from_list(
            "isapn_score",
            [(0, "orangered"), (0.5, "white"), (1, "royalblue")],
        )

        def draw(axis, values, title, scale=1):
            png = showAtomHighlight(
                molecule,
                values,
                color_map,
                atom_with_index,
                rot,
                line_width,
                figsize=scale,
            )
            axis.imshow(Image.open(io.BytesIO(png)))
            axis.axis("off")
            axis.set_title(title, fontsize=20)

        if only_total:
            figure, axis = plt.subplots(
                1,
                1,
                figsize=(16 * figsize, 11 * figsize),
            )
            draw(axis, total, " ", figsize)
        else:
            if locate == "right":
                grid = GridSpec(2, 3)
                positions = (grid[0:2, 0:2], grid[0, 2], grid[1, 2])
                figure = plt.figure(figsize=(16 * figsize, 10 * figsize))
            elif locate == "bottom":
                grid = GridSpec(3, 2)
                positions = (grid[0:2, 0:2], grid[2, 0], grid[2, 1])
                figure = plt.figure(figsize=(12 * figsize, 12 * figsize))
            else:
                raise ValueError("locate must be either 'right' or 'bottom'.")
            axes = [figure.add_subplot(position) for position in positions]
            draw(axes[0], total, " ", figsize)
            draw(axes[1], positive, "Positive", figsize * 0.5)
            draw(axes[2], negative, "Negative", figsize * 0.5)
            axis = axes[0]
        if with_colorbar:
            colorbar = plt.colorbar(
                plt.cm.ScalarMappable(cmap=color_map),
                ax=axis,
                shrink=0.5,
                pad=0,
            )
            colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
            labels = ticks
            if tick_type == "minmax":
                labels = np.round(
                    np.linspace(np.min(raw_total), np.max(raw_total), 5),
                    3,
                )
            colorbar.set_ticklabels(labels)
        plt.show()
        return {
            "positive": positive,
            "negative": negative,
            "total": total,
        }

    def get_subgroup_score_bin(self, smiles_list):
        """Group both ISATPN score streams by fragment SMILES."""

        positive_fragments = {}
        negative_fragments = {}
        for smiles, result in self.get_scores(smiles_list).items():
            positive = np.asarray(result["positive"]).reshape(-1, 1)
            negative = np.asarray(result["negative"]).reshape(-1, 1)
            for index, fragment in enumerate(self.get_fragment(smiles)):
                for destination, values in (
                    (positive_fragments, positive),
                    (negative_fragments, negative),
                ):
                    current = values[index]
                    if fragment.smiles in destination:
                        current = np.concatenate(
                            [destination[fragment.smiles], current],
                            axis=0,
                        )
                    destination[fragment.smiles] = current
        return positive_fragments, negative_fragments

    def plot_score_hist(self, smiles_list, xlim=(0, 1), bins=40):
        """Plot and return both ISATPN score streams."""

        positive = []
        negative = []
        for result in self.get_scores(smiles_list).values():
            positive.extend(set(np.asarray(result["positive"]).flatten()))
            negative.extend(set(np.asarray(result["negative"]).flatten()))
        edges = np.arange(xlim[0], xlim[1], 1.0 / bins)
        plt.hist(positive, bins=edges, alpha=0.5, label="positive")
        plt.hist(negative, bins=edges, alpha=0.5, label="negative")
        plt.legend()
        plt.xlim(*xlim)
        plt.show()
        return positive, negative

    def plot_subgroup_score_histogram(
        self,
        smiles_list,
        nums=10,
        bins=40,
        xlim=(0, 1),
    ):
        """Plot fragment histograms separately for both ISATPN streams."""

        positive, negative = self.get_subgroup_score_bin(smiles_list)
        positive = dict(
            sorted(positive.items(), key=lambda item: len(item[1]), reverse=True)[
                :nums
            ]
        )
        negative = dict(
            sorted(negative.items(), key=lambda item: len(item[1]), reverse=True)[
                :nums
            ]
        )
        edges = np.arange(xlim[0], xlim[1], 1.0 / bins)
        for title, values in (("Positive", positive), ("Negative", negative)):
            for fragment, scores in values.items():
                plt.hist(scores, bins=edges, alpha=0.5, label=fragment)
            plt.xlim(*xlim)
            plt.legend()
            plt.title(title)
            plt.show()
        return positive, negative

    def plot_subgroup_score_histogram_byone(
        self,
        smiles_list,
        nums=10,
        bins=40,
        xlim=(0, 1),
        target_subgroups=None,
    ):
        """Plot both ISATPN streams together for each selected fragment."""

        positive, negative = self.get_subgroup_score_bin(smiles_list)
        positive = dict(
            sorted(positive.items(), key=lambda item: len(item[1]), reverse=True)[
                :nums
            ]
        )
        negative = dict(
            sorted(negative.items(), key=lambda item: len(item[1]), reverse=True)[
                :nums
            ]
        )
        if target_subgroups:
            positive = {
                key: value
                for key, value in positive.items()
                if key in target_subgroups
            }
            negative = {
                key: value
                for key, value in negative.items()
                if key in target_subgroups
            }
        figure, axes = plt.subplots(
            max(len(positive), 1),
            1,
            figsize=(5, 2 * max(len(positive), 1)),
            squeeze=False,
        )
        edges = np.arange(xlim[0], xlim[1], 1.0 / bins)
        for axis, (fragment, values) in zip(axes[:, 0], positive.items()):
            axis.hist(values, bins=edges, alpha=0.5, label="positive")
            if fragment in negative:
                axis.hist(
                    negative[fragment],
                    bins=edges,
                    alpha=0.5,
                    label="negative",
                )
            axis.set_ylabel(fragment)
            axis.set_xlim(*xlim)
            axis.legend()
        plt.show()
        return positive, negative


class ISAPNAnalyzer(_ISAPNAnalyzerMixin, ISAAnalyzer):
    """Legacy-compatible Analyzer for ISATPN positive/negative models."""

    def __init__(self, model_path, save_result=False, **kwargs):
        super().__init__(model_path, save_result, **kwargs)
        self._configure_isapn_interpreter()


class ISAPNAnalyzer_v2(_ISAPNAnalyzerMixin, ISAAnalyzer_v2):
    """Version 2 Analyzer for ISATPN positive/negative models."""

    def __init__(self, model_path, save_result=False, **kwargs):
        super().__init__(model_path, save_result, **kwargs)
        self._configure_isapn_interpreter()


class ISAPNAnalyzer_v1p3(ISAPNAnalyzer_v2):
    """Deprecated compatibility name for :class:`ISAPNAnalyzer_v2`."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ISAPNAnalyzer_v1p3 is deprecated; use ISAPNAnalyzer_v2 or "
            "Analyzer(...) instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


ISATPNAnalyzer = ISAPNAnalyzer
ISATPNAnalyzer_v2 = ISAPNAnalyzer_v2

__all__ = [
    "ISAPNAnalyzer",
    "ISAPNAnalyzer_v2",
    "ISAPNAnalyzer_v1p3",
    "ISATPNAnalyzer",
    "ISATPNAnalyzer_v2",
]
