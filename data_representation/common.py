from dataclasses import dataclass, field
from math import ceil

from music21.instrument import Instrument
from music21.pitch import Pitch
from music21.duration import Duration

PITCHES = list(range(-1, 129))
TIME_SHIFTS = [i/100 for i in range(1, 101)]
TEMPOS = list(range(1, 300))

@dataclass
class Symbol:
    instrument:str
    pitch:int
    duration:float
    offset:float

    def get(
            instrument:Instrument, 
            pitch:Pitch|None, 
            duration:Duration, 
            offset:float
        ):
        instrument = instrument.__class__.__name__
        duration = ceil(duration.quarterLength * 100) / 100
        offset = ceil(offset * 100) / 100

        if pitch:
            pitch = pitch.midi
        else:
            pitch = -1

        return Symbol(instrument, pitch, duration, offset)

Piece = list[Symbol]


class Token:
    pass

@dataclass
class NoteOn(Token):
    pitch:int

@dataclass
class NoteOff(Token):
    pitch:int

@dataclass
class TimeShift(Token):
    cs:float

@dataclass
class Instrument(Token):
    instrument:str

class Tempo(Token):
    number:int

@dataclass
class Start(Token):
    pass

@dataclass
class End(Token):
    pass
