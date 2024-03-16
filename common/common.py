from dataclasses import dataclass

from torch import device

from common.symbol import Symbol

PITCHES = list(range(-1, 129))
TIME_SHIFTS = list(range(1, 101))
TEMPOS = list(range(20, 221))

DEVICE = device("cpu")