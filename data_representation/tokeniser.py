from math import floor

from data_representation.common import (
    Symbol, Piece, 
    Token, NoteOn, NoteOff, TimeShift, 
    Instrument, Start, End
)

def encode_tokens(symbols:list[Symbol]) -> list[(float, list[Token])]:
    encoded_symbols = {}

    for symbol in symbols:
        instrument = symbol.instrument
        pitch = symbol.pitch
        on_offset = symbol.offset
        off_offset = on_offset + symbol.duration

        if not on_offset in encoded_symbols:
            encoded_symbols[on_offset] = []
        if not off_offset in encoded_symbols:
            encoded_symbols[off_offset] = []

        encoded_symbols[on_offset] += [Instrument(instrument), NoteOn(pitch)]
        encoded_symbols[off_offset] += [Instrument(instrument), NoteOff(pitch)]

    return sorted(encoded_symbols.items())

def get_time_shift_sequence(prev_offset:float, next_offset:float) -> list[float]:
    shift_duration = round(next_offset - prev_offset, 2)
    seconds = floor(shift_duration)
    
    time_shift_sequence = [TimeShift(1.0) for _ in range(seconds)]

    if seconds != shift_duration:
        time_shift_sequence.append(TimeShift(shift_duration % 1))

    return time_shift_sequence

def insert_time_shifts(timed_tokens:list[(float, list[Token])]) -> list[Token]:
    prev_offset = timed_tokens[0][0]
    tokens = get_time_shift_sequence(0, prev_offset) + timed_tokens[0][1]

    for (offset, tk) in timed_tokens[1:]:
        tokens += get_time_shift_sequence(prev_offset, offset)
        tokens += tk

        prev_offset = offset

    return tokens

def remove_instrument_duplicates(tokens: list[Token]) -> list[Token]:
    filtered_tokens = []
    prev_instrument = None

    for token in tokens:
        if (isinstance(token, Instrument)):
            if token.instrument == prev_instrument:
                continue
            prev_instrument = token.instrument

        filtered_tokens.append(token)

    return filtered_tokens

def piece_to_tokens(piece: Piece) -> list[Token]:
    timed_tokens = encode_tokens(piece)
    unfiltered_tokens = insert_time_shifts(timed_tokens)
    tokens = remove_instrument_duplicates(unfiltered_tokens)

    return [Start()] + tokens + [End()]
