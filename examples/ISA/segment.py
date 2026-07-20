"""Inspect the default ISA fragmentation of a small molecule."""

from D4CMPP2 import Segmentator


segmentator = Segmentator(6, 2, 0)
groups = segmentator.segment("CCOC(=O)c1ccccc1", get_only_index=True)
print(groups)
