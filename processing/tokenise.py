from math import floor

from common.token import (
    NoteOn, NoteOff, Tempo,
    Instrument, TimeShift,
    Start, End
)

def _encode_tokens(symbols):
    encoded_symbols = {}

    for symbol in symbols:
        instrument = symbol.instrument
        pitch = symbol.pitch
        on_offset = symbol.offset
        off_offset = on_offset + symbol.duration
        tempo = symbol.tempo

        if not on_offset in encoded_symbols:
            encoded_symbols[on_offset] = []
        if not off_offset in encoded_symbols:
            encoded_symbols[off_offset] = []

        encoded_symbols[on_offset] += [Instrument(instrument), Tempo(tempo), NoteOn(pitch)]
        encoded_symbols[off_offset] += [Instrument(instrument), Tempo(tempo), NoteOff(pitch)]

    return sorted(encoded_symbols.items())

def _get_time_shift_sequence(prev_offset, next_offset):
    shift_duration = next_offset - prev_offset
    seconds = floor(shift_duration/100)
    
    time_shift_sequence = [TimeShift(100) for _ in range(seconds)]

    if shift_duration % 100 != 0:
        time_shift_sequence.append(TimeShift(shift_duration % 100))

    return time_shift_sequence

def _insert_time_shifts(timed_tokens):
    prev_offset = timed_tokens[0][0]
    tokens = _get_time_shift_sequence(0, prev_offset) + timed_tokens[0][1]

    for (offset, tk) in timed_tokens[1:]:
        tokens += _get_time_shift_sequence(prev_offset, offset)
        tokens += tk

        prev_offset = offset

    return tokens

def _remove_duplicates(tokens):
    filtered_tokens = []

    prev_instrument = None
    prev_tempo = None

    for token in tokens:
        if (isinstance(token, Instrument)):
            if token.instrument == prev_instrument: continue
            prev_instrument = token.instrument

        elif (isinstance(token, Tempo)):
            if token.tempo == prev_tempo: continue
            prev_tempo = token.tempo

        filtered_tokens.append(token)

    return filtered_tokens


def tokenise(symbols):
    timed_tokens = _encode_tokens(symbols)
    unfiltered_tokens = _insert_time_shifts(timed_tokens)
    tokens = _remove_duplicates(unfiltered_tokens)

    return [Start()] + tokens + [End()]