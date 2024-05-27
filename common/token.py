from dataclasses import dataclass

from common.common import (
    PITCHES, TIME_SHIFTS, TEMPOS
)

class Token:
    pass

@dataclass
class NoteOn(Token):
    pitch:int

    def __init__(self, pitch):
        '''
        Initialise the NoteOn token.

        Parameters:
            pitch (int): Pitch of the note.
        '''
        pitch = max(pitch, PITCHES[0])
        pitch = min(pitch, PITCHES[-1])
        self.pitch = int(pitch)

@dataclass
class NoteOff(Token):
    pitch:int

    def __init__(self, pitch):
        '''
        Initialise the NoteOff token.

        Parameters:
            pitch (int): Pitch of the note.
        '''
        pitch = max(pitch, PITCHES[0])
        pitch = min(pitch, PITCHES[-1])
        self.pitch = int(pitch)

@dataclass
class TimeShift(Token):
    cs:int

    def __init__(self, cs):
        '''
        Initialise the TimeShift token.

        Parameters:
            cs (int): Duration of the time shift in centiseconds.
        '''
        cs = max(cs, TIME_SHIFTS[0])
        cs = min(cs, TIME_SHIFTS[-1])
        self.cs = int(cs)

@dataclass
class Tempo(Token):
    tempo:int

    def __init__(self, tempo):
        '''
        Initialise the Tempo token.

        Parameters:
            tempo (int): The tempo (beats per minute) of the notes.
        '''
        tempo = max(tempo, TEMPOS[0])
        tempo = min(tempo, TEMPOS[-1])
        self.tempo = int(tempo)

@dataclass
class Instrument(Token):
    instrument:str

@dataclass
class Start(Token):
    pass

@dataclass
class End(Token):
    pass
