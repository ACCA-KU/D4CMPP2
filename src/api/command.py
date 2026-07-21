"""Command-line entry point using the same training path as the Python API."""

from D4CMPP2.src.api.training import train


def main():
    train(use_argparser=True)
