from math import floor

from common.token import (
    NoteOn, NoteOff, Tempo,
    Instrument, TimeShift,
    Start, End
)

def _encode_tokens(symbols):
    '''
    Partition a symbol sequence into token sets partitioned by the offset the tokens are triggered on.

    Parameters:
        symbols (list[Symbol]): The symbols.

    Returns:
        timed_tokens (list[tuple[int, list[Token]]]): The partitioned tokens.
    '''
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

    timed_tokens = sorted(encoded_symbols.items())

    return timed_tokens

def _get_time_shift_sequence(prev_offset, next_offset):
    '''
    Get sequence of time shifts that represent the change in time between the previous and next offsets.

    Parameters:
        prev_offset (int): The previous offset.
        next_offset (int): The next offset.

    Returns:
        time_shift_sequence (list[TimeShift]): The time shift sequence.
    '''
    shift_duration = next_offset - prev_offset
    seconds = floor(shift_duration/100)
    
    time_shift_sequence = [TimeShift(100) for _ in range(seconds)]

    if shift_duration % 100 != 0:
        time_shift_sequence.append(TimeShift(shift_duration % 100))

    return time_shift_sequence

def _insert_time_shifts(timed_tokens):
    '''
    Create a token sequence from token sets partitioned by their offset times.

    Parameters:
        timed_tokens (list[tuple[int, list[Token]]]): The partitioned tokens.

    Returns:
        tokens (list[Token]): The token sequence.
    '''
    prev_offset = timed_tokens[0][0]
    tokens = _get_time_shift_sequence(0, prev_offset) + timed_tokens[0][1]

    for (offset, tk) in timed_tokens[1:]:
        tokens += _get_time_shift_sequence(prev_offset, offset)
        tokens += tk

        prev_offset = offset

    return tokens

def _remove_duplicates(tokens):
    '''
    Filter out duplicate instrument and tempo token sequences and returns the filtered tokens.

    Parameters:
        tokens (list[Token]): The tokens.

    Returns:
        filtered_tokens (list[Token]): The filtered tokens.
    '''
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
    '''
    Convert symbols into token objects.

    Parameters:
        symbols (list[Symbol]): The symbols.

    Returns:
        tokens (list[Token]): The tokens.
    '''
    timed_tokens = _encode_tokens(symbols)
    unfiltered_tokens = _insert_time_shifts(timed_tokens)
    tokens = _remove_duplicates(unfiltered_tokens)

    tokens = [Start()] + tokens + [End()]

    return tokens