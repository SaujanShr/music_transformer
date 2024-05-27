from dataclasses import dataclass
from re import findall

def _pad_camel_case(s):
    '''
    Space out the camel-case-seperated words in the string.

    Parameters:
        s (str): Camel-case string.

    Returns:
        spaced (str): Spaced out string.
    '''
    words = findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', s)
    spaced = ' '.join(words)
    
    return spaced


@dataclass
class Symbol:
    instrument:str
    pitch:int
    duration:int
    tempo:int
    offset:int

    def build(instrument, pitch, duration, tempo, offset):
        '''
        Build a symbol object from music21 objects.

        Parameters:
            instrument (Instrument): The music21 instrument object.
            pitch (Pitch): The music21 pitch object.
            duration (Duration): The music21 duration object.
            tempo (MetronomeMark): The music21 MetronomeMark object.
            offset (OffsetQL): The music21 offset value.

        Returns:
            symbol (Symbol): The built symbol.
        '''
        instrument = _pad_camel_case(instrument.__class__.__name__)
        
        if pitch: pitch = pitch.midi
        else: pitch = -1

        duration = round(duration.quarterLength*100)
        tempo = round(tempo.number, -1)
        offset = round(offset*100)

        symbol = Symbol(instrument, pitch, duration, tempo, offset)

        return symbol
