import argparse


SUPPORTED_IDS = (
    "GCN", "GCNwS", "MPNN", "MPNNwS", "DMPNN", "DMPNNwS",
    "AFP", "AFPwS", "GAT", "GATwS", "GC", "ISAT", "ISATPN",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="d4cmpp2",
        description="Train or continue a D4CMPP2 molecular property model.",
        epilog="Supported network IDs: " + ", ".join(SUPPORTED_IDS),
    )
    parser.add_argument("-d", "--data", help="CSV path or packaged dataset name")
    parser.add_argument("-t", "--target", help="Comma-separated target columns, e.g. Solubility,LogP")
    parser.add_argument("-n", "--network", choices=SUPPORTED_IDS, help="Network ID")
    parser.add_argument("-l", "--load", dest="LOAD_PATH", help="Saved model directory to continue")
    parser.add_argument("--resume", dest="RESUME_PATH", help="Saved model directory or .ckpt for full-state resume")
    parser.add_argument("--transfer", dest="TRANSFER_PATH", help="Saved model directory for transfer learning")
    parser.add_argument("-f", "--fragment", help="ISA sculptor index as three comma-separated integers, e.g. 6,2,0")
    parser.add_argument("--seed", dest="random_seed", type=int, help="Seed Python, NumPy, torch, CUDA, shuffle, and partial training")
    parser.add_argument("--deterministic", dest="deterministic_algorithms", action="store_true", default=None, help="Enable PyTorch deterministic algorithms; requires --seed")
    parser.add_argument("-e", "--epoch", dest="max_epoch", type=int, help="Maximum training epochs")
    parser.add_argument("--device", default=None, help="Training device, e.g. cpu or cuda:0")
    parser.add_argument("-c", "--cuda", type=int, help="Deprecated shorthand for --device cuda:N")
    parser.add_argument("--partial-train", dest="partial_train", type=float, help="Training fraction (0..1) or row count")
    parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        default=None,
        help="Hide informational messages and progress bars; warnings and errors remain visible",
    )

    args = parser.parse_args(argv)
    if args.target:
        args.target = [value.strip() for value in args.target.split(",") if value.strip()]
    if args.cuda is not None:
        if args.device is not None:
            parser.error("use either --device or --cuda, not both")
        args.device = f"cuda:{args.cuda}"
    del args.cuda
    if args.partial_train is not None and args.partial_train > 1:
        args.partial_train = int(args.partial_train)
    if args.fragment:
        try:
            values = tuple(int(value.strip()) for value in args.fragment.split(","))
        except ValueError as exc:
            parser.error(f"--fragment must contain integers: {exc}")
        if len(values) != 3 or any(value < 0 for value in values):
            parser.error("--fragment must be three non-negative integers, e.g. 6,2,0")
        args.sculptor_index = values
    del args.fragment
    return args
