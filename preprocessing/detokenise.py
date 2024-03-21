from common.common import PITCHES
from common.symbol import Symbol
from common.token import (
    Instrument, TimeShift,
    Tempo, NoteOn, NoteOff
)

def _get_instruments(tokens):
    '''
    Collect all the unique instruments in the tokens.

    Parameters:
        tokens (list[Token]): List of tokens

    Returns:
        instruments (set[str]): Set of unique instrument names
    '''
    instruments = {
        token.instrument
        for token in tokens
        if isinstance(token, Instrument)
    }

    return instruments


def detokenise(tokens):
    """
    Transform a list of tokens into a list of symbols.

    Parameters:
        tokens (list[Token]): List of tokens

    Returns:
        symbols (list[Symbol]): List of symbols
    """
    symbols = []

    instruments = _get_instruments(tokens)

    instrument_notes = {
        instrument: {
            pitch: [] 
            for pitch in PITCHES
        }
        for instrument in instruments
    }

    offset = 0

    curr_tempo = 100
    curr_instrument = next(iter(instruments))
    curr_instrument_notes = instrument_notes[curr_instrument]

    for token in tokens:
        if isinstance(token, Instrument):
            curr_instrument = token.instrument
            curr_instrument_notes = instrument_notes[curr_instrument]
        
        elif isinstance(token, TimeShift):
            offset += token.cs

        elif isinstance(token, Tempo):
            curr_tempo = token.tempo
        
        elif isinstance(token, NoteOn):
            curr_instrument_notes[token.pitch].append(offset)

        elif isinstance(token, NoteOff):
            if curr_instrument_notes[token.pitch]:
                on_offset = curr_instrument_notes[token.pitch][0]
                off_offset = offset

                instrument = curr_instrument
                pitch = token.pitch
                duration = off_offset - on_offset
                tempo = curr_tempo

                symbols.append(Symbol(instrument, pitch, duration, tempo, on_offset))

                curr_instrument_notes[token.pitch].pop(0)
    
    symbols = sorted(symbols, key=lambda symbol:symbol.offset)

    return symbols