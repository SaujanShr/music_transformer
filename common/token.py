from dataclasses import dataclass
from decimal import Decimal

from common.common import (
    PITCHES, TIME_SHIFTS, TEMPOS
)

class Token:
    pass

@dataclass
class NoteOn(Token):
    pitch:int

    def __init__(self, pitch):
        pitch = max(pitch, PITCHES[0])
        pitch = min(pitch, PITCHES[-1])
        self.pitch = pitch

@dataclass
class NoteOff(Token):
    pitch:int

    def __init__(self, pitch):
        pitch = max(pitch, PITCHES[0])
        pitch = min(pitch, PITCHES[-1])
        self.pitch = pitch

@dataclass
class TimeShift(Token):
    cs:Decimal

    def __init__(self, cs):
        cs = max(cs, TIME_SHIFTS[0])
        cs = min(cs, TIME_SHIFTS[-1])
        self.cs = cs

@dataclass
class Instrument(Token):
    instrument:str

@dataclass
class Tempo(Token):
    tempo:int

    def __init__(self, tempo):
        tempo = max(tempo, TEMPOS[0])
        tempo = min(tempo, TEMPOS[-1])
        self.tempo = tempo

@dataclass
class Start(Token):
    pass

@dataclass
class End(Token):
    pass
