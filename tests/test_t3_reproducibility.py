import random
import unittest
from pathlib import Path

from fixtures import TINY_REGRESSION_CSV, isolated_workdir
from markers import heavy_test


class ReproducibilityPolicyTests(unittest.TestCase):
    @heavy_test
    def test_seed_repeats_python_numpy_torch_and_resume_does_not_reseed(self):
        import numpy as np
        import torch

        from D4CMPP2.src.utils.reproducibility import configure_reproducibility

        config = {"random_seed": 123, "split_random_seed": 42}
        configure_reproducibility(config)
        first = (
            random.random(),
            float(np.random.random()),
            torch.rand(3),
            torch.nn.Linear(2, 2).weight.detach().clone(),
        )
        configure_reproducibility(config)
        second = (
            random.random(),
            float(np.random.random()),
            torch.rand(3),
            torch.nn.Linear(2, 2).weight.detach().clone(),
        )
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertTrue(torch.equal(first[3], second[3]))

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        configure_reproducibility(config, resume=True)
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(np.random.get_state()[1].tolist(), numpy_state[1].tolist())
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    @heavy_test
    def test_same_seed_repeats_cpu_smoke_training(self):
        import pandas as pd
        import torch

        from D4CMPP2 import train

        with isolated_workdir() as root:
            (root / "models").mkdir()
            (root / "graphs").mkdir()
            outputs = []
            for name in ("first", "second"):
                model_path = root / name
                result = train(
                        data=str(TINY_REGRESSION_CSV),
                        target=["target_a"],
                        network="GCN",
                        MODEL_PATH=str(model_path),
                        MODEL_DIR=str(root / "models"),
                        GRAPH_DIR=str(root / "graphs"),
                        device="cpu",
                        pin_memory=False,
                        random_seed=31415,
                        split_random_seed=42,
                        max_epoch=1,
                        batch_size=4,
                        hidden_dim=8,
                        conv_layers=2,
                        linear_layers=2,
                        dropout=0.1,
                        save_prediction=False,
                )
                curve = pd.read_csv(Path(result) / "result" / "learning_curve.csv")
                weights = torch.load(
                    Path(result) / "final.pth",
                    map_location="cpu",
                    weights_only=True,
                )
                outputs.append((curve, weights))

        pd.testing.assert_frame_equal(outputs[0][0], outputs[1][0], check_exact=True)
        self.assertEqual(outputs[0][1].keys(), outputs[1][1].keys())
        for key in outputs[0][1]:
            self.assertTrue(torch.equal(outputs[0][1][key], outputs[1][1][key]), key)
