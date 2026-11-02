"""Key handling for the waterline labeling tool.

The tool must work with any keyboard state (Caps Lock, non-English layout),
so next/redo/skip/quit also have layout-independent keys.
"""

import pytest

from swimsync.waterline.label import key_action


@pytest.mark.parametrize("key, action", [
    (ord("n"), "next"), (ord("N"), "next"), (13, "next"), (32, "next"),
    (ord("r"), "redo"), (ord("R"), "redo"), (8, "redo"), (127, "redo"),
    (ord("s"), "skip"), (ord("S"), "skip"),
    (ord("q"), "quit"), (ord("Q"), "quit"), (27, "quit"),
])
def test_key_action(key, action):
    assert key_action(key) == action


@pytest.mark.parametrize("key", [-1, 255, 0x5DE])  # no key, no key (masked), Hebrew mem
def test_other_keys_do_nothing(key):
    assert key_action(key) is None
