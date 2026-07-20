import hashlib
import json
import unittest

from markers import heavy_test


EXPECTED_SIGNATURES = {
    "afp": "d473d5e1e8d6000bace3ce143f4d2b7287fadce91df751df779a433be72bbfac",
    "afp_solvent": "7b1bdf47c0255c51a03ad555729d311cbd45249f2a14b056543d6200a0f4a400",
    "dmpnn": "5639f81a8625e51337f2f31e99cf49e74810e79a7d8e24cf3dc47cc6129cadd3",
    "dmpnn_solvent": "a2a84e45ae4bb788d6fbf8a3f710dbb6b7f16dc33fa74cf7b9b1df9db83fa919",
    "gat": "6900d70c771e66ebc458618a39a90e1f106ddefa3e31858265a0caffe4a4d0eb",
    "gat_solvent": "a60b34cbd0dec38e2e288f9ece469be38b3cdeaaaeec656cdbaec87d2956d622",
    "gcn": "d4d1d4d9b7712189ecbab6e60108f3312219181ceaa4e3dae3ac294ce0ecc2bf",
    "gcn_solvent": "1cd694f9532ee3cf9768c07bb5b9c9e180d98bdf1b5a87286e7595232b4081a4",
    "group_contribution": "84b4c88f4e47ec2672ca14a4bbcdafdd6a735819188787c54906f83caaa38563",
    "isat": "9e7a1287d0d0c1ddec8e4846added12cf45c29d86a94f38b00afe92ef6c31565",
    "isatpn": "4db915f8e29d93dbca8898392dacdbf0cb04f4dd11de989979b6dcff75997f6d",
    "mpnn": "ce39b0cbf22bc112d995f3072f4d772ae997bf262c08278f2529cb0d40417ab6",
    "mpnn_solvent": "bc84be22a44400efc26719594014844fbbb77f80cb78641481aa98ae1d6e0eee",
}


class StateDictContractTests(unittest.TestCase):
    @heavy_test
    def test_all_builtin_parameter_keys_and_shapes_match_snapshot(self):
        from D4CMPP2.networks.registry import registered_models

        config = {
            "node_dim": 36,
            "edge_dim": 12,
            "target_dim": 1,
            "hidden_dim": 16,
            "conv_layers": 1,
            "linear_layers": 1,
            "dropout": 0.0,
        }
        observed = {}
        for name, definition in sorted(registered_models().items()):
            if name not in EXPECTED_SIGNATURES:
                continue
            state = [
                (key, list(value.shape))
                for key, value in definition.network(config).state_dict().items()
            ]
            observed[name] = hashlib.sha256(
                json.dumps(state, separators=(",", ":")).encode()
            ).hexdigest()
        self.assertEqual(observed, EXPECTED_SIGNATURES)


if __name__ == "__main__":
    unittest.main()
