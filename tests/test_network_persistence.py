import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from markers import heavy_test


class NetworkPersistenceTests(unittest.TestCase):
    @heavy_test
    def test_saved_network_is_source_snapshot_with_identity(self):
        from D4CMPP2.networks.GCN_model import GCN
        from D4CMPP2.src.utils.supportfile_saver import save_registered_network

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-network-snapshot-") as temporary:
            root = Path(temporary)
            save_registered_network(GCN, str(root))
            saved = (root / "network.py").read_text(encoding="utf-8")
            identity = json.loads(
                (root / "network_identity.json").read_text(encoding="utf-8")
            )
            source = Path(inspect.getsourcefile(GCN)).read_text(encoding="utf-8")

            self.assertIn("class GCN(MolecularNetwork)", saved)
            self.assertIn("network = GCN", saved)
            self.assertEqual(identity["class"], "GCN")
            self.assertEqual(
                identity["source_sha256"],
                hashlib.sha256(source.encode("utf-8")).hexdigest(),
            )

    @heavy_test
    def test_saved_snapshot_precedes_current_registry(self):
        from D4CMPP2.src.NetworkManager.NetworkManager import NetworkManager

        source = """
class SavedNetwork:
    source = "saved-snapshot"

network = SavedNetwork
"""
        with tempfile.TemporaryDirectory(prefix="d4cmpp2-network-priority-") as temporary:
            root = Path(temporary)
            (root / "network.py").write_text(source, encoding="utf-8")
            manager = NetworkManager.__new__(NetworkManager)
            manager.config = {
                "MODEL_PATH": str(root),
                "network_id": "gcn",
            }

            selected = manager.get_net_module()
            transfer_selected = manager.get_net_module(str(root))
            self.assertEqual(selected.source, "saved-snapshot")
            self.assertEqual(transfer_selected.source, "saved-snapshot")

    @heavy_test
    def test_solvent_aliases_and_canonical_registry_id(self):
        from D4CMPP2._main import load_NET_REFER
        from D4CMPP2.networks.GCNwithSolv_model import SolventGCN
        from fixtures import ROOT

        model = SolventGCN(
            {
                "node_dim": 8,
                "target_dim": 1,
                "hidden_dim": 8,
                "conv_layers": 1,
                "linear_layers": 1,
                "dropout": 0.0,
                "solv_hidden_dim": 24,
                "solv_conv_layers": 3,
                "solv_linear_layers": 2,
            }
        )
        self.assertEqual(model.config["solvent_hidden_dim"], 24)
        self.assertEqual(model.config["solvent_conv_layers"], 3)
        self.assertEqual(model.config["solvent_linear_layers"], 2)
        with self.assertRaisesRegex(ValueError, "conflicting values"):
            SolventGCN(
                {
                    "node_dim": 8,
                    "target_dim": 1,
                    "solv_hidden_dim": 24,
                    "solvent_hidden_dim": 32,
                }
            )

        resolved = load_NET_REFER(
            {
                "network": "GCNwS",
                "NET_REFER": str(ROOT / "network_refer.yaml"),
            }
        )
        self.assertEqual(resolved["network_id"], "gcn_solvent")


if __name__ == "__main__":
    unittest.main()
