from music21.stream import Stream, Part
from music21.instrument import fromString, Cowbell, Piano
from music21.note import Note, Unpitched
from music21.pitch import Pitch
from music21.duration import Duration
from music21.tempo import MetronomeMark

def _get_instrument(instrument):
    '''
    Attempt to get a music21 instrument from the string.
    If it doesn't exist, return a piano by default.

    Parameters:
        instrument (str): The name of the instrument.
    
    Returns:
        instrument (Instrument): The instrument object.
    '''
    if instrument == "Cowbell":
        return Cowbell()
    try:
        instrument = fromString(instrument)
    except Exception:
        instrument = Piano()

    return instrument


def _partition_by_instrument(symbols):
    '''
    Partition the symbols by the the symbol's instrument.

    Parameters:
        symbols (list[Symbol]): The symbols.

    Returns:
        parts (list[tuple[str, list[Symbol]]]): The instrument partitions.
    '''
    partitions = {}

    for symbol in symbols:
        instrument = symbol.instrument

        if instrument not in partitions:
            partitions[instrument] = []
        partitions[instrument].append(symbol)

    parts = [
        (_get_instrument(instrument), part)
        for (instrument, part) in partitions.items()
    ]

    return parts

def _get_composition(parts):
    '''
    Create a music composition from the given instrument partition.

    Parameters:
        parts (list[tuple[str, list[Symbol]]]): The instrument partitions.

    Returns:
        composition (Stream): The composition.
    '''
    composition = Stream()

    for (instrument, symbols) in parts:
        part = Part()
        composition.append(part)

        part.offset = 0
        part.append(instrument)

        tempo = float('-inf')
        for symbol in symbols:
            if tempo != symbol.tempo:
                tempo = symbol.tempo
                part.append(MetronomeMark(tempo))

            if symbol.pitch == -1:
                note = Unpitched()
            else:
                note = Note()
                note.pitch = Pitch(midi=symbol.pitch)

            part.append(note)
            note.duration = Duration(symbol.duration/100)
            note.offset = symbol.offset/100

    return composition

def midi(symbols, filename):
    '''
    Create a MIDI file from the symbols and save it with the filename.

    Parameters:
        symbols (list[Symbol]): The symbols.
        filename (str): The name of the MIDI file.
    '''
    parts = _partition_by_instrument(symbols)

    composition = _get_composition(parts)

    composition.write('midi', fp=f'bin/midi/{filename}.mid')