"""Command-line entry point using the same training path as the Python API."""

from D4CMPP2._main import train


def main():
    train(use_argparser=True)
