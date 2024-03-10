from math import ceil

from music21.instrument import Instrument
from music21.pitch import Pitch
from music21.duration import Duration

from data_representation.common import (
    Symbol, Token, PITCHES,
    NoteOn, NoteOff,
    TimeShift, Instrument
)

def tokens_to_piece(tokens:list[Token], instruments:set[str]):
    symbols = []

    instrument_notes = { 
        instrument:{ pitch:[] for pitch in PITCHES } 
        for instrument in instruments 
    }
    offset = 0.0
    curr_instrument = next(iter(instruments))
    curr_instrument_notes = instrument_notes[curr_instrument]

    for token in tokens:
        if isinstance(token, Instrument):
            curr_instrument = token.instrument
            curr_instrument_notes = instrument_notes[curr_instrument]
        
        elif isinstance(token, TimeShift):
            offset += token.cs
        
        elif isinstance(token, NoteOn):
            curr_instrument_notes[token.pitch].append(offset)

        elif isinstance(token, NoteOff):
            if curr_instrument_notes[token.pitch]:
                pitch = token.pitch

                instrument = curr_instrument
                on_offset = curr_instrument_notes[pitch][0]
                off_offset = offset
                duration = round(off_offset - on_offset, 2)

                symbols.append(Symbol(curr_instrument, pitch, duration, on_offset))

                curr_instrument_notes[token.pitch].pop(0)
    
    piece = sorted(symbols, key=lambda symbol:symbol.offset)

    return piece