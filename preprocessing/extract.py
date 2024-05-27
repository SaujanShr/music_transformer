from pathlib import Path

from tqdm.contrib.concurrent import thread_map
from itertools import chain
from random import shuffle

from music21.converter import parse
from music21.instrument import Piano, partitionByInstrument
from music21.chord import Chord
from music21.percussion import PercussionChord
from music21.note import Note, Unpitched
from music21.tempo import MetronomeMark

from common.symbol import Symbol

def _part_to_symbols(part):
    '''
    Extract symbols from the stream partition.
    
    If the stream contains more than one instrument, 
    then only the first instrument is counted.

    Parameters:
        part (Stream): The stream partition.

    Returns:
        symbols (list[Symbol]): The symbols.
    '''
    if part.getInstrument() == None:
        instrument = Piano()
    else:
        instrument = part.getInstrument()
    
    tempo = MetronomeMark(number=100)
    symbols = []

    for element in part.flatten():
        if isinstance(element, Chord):
            symbols += [
                Symbol.build(instrument, note.pitch, note.duration, tempo, element.offset + note.offset)
                for note in element.notes
            ]
        elif isinstance(element, PercussionChord):
            symbols += [
                Symbol.build(note.storedInstrument, None, note.duration, tempo, element.offset + note.offset)
                for note in element.notes
            ]
        elif isinstance(element, Note):
            symbols.append(
                Symbol.build(instrument, element.pitch, element.duration, tempo, element.offset)
            )
        elif isinstance(element, Unpitched):
            symbols.append(
                Symbol.build(element.storedInstrument, None, element.duration, tempo, element.offset)
            )
        elif isinstance(element, MetronomeMark):
            tempo = element
    
    return symbols


def _get_symbols(file):
    '''
    Parse the MIDI file to a stream and extract symbols from the stream.

    Parameters:
        file (Path): The path to the MIDI file.

    Returns:
        symbols (list[Symbol]): The symbols.
    '''
    midi = parse(file)
    parts = partitionByInstrument(midi)

    part_symbols = [_part_to_symbols(part) for part in parts]
    symbols = list(chain.from_iterable(part_symbols))
    symbols = sorted(symbols, key=lambda symbol:symbol.offset)

    return symbols


def extract(genre, sample_size=None):
    '''
    Parse the MIDI files from the given genre of music
    and extract symbols for each MIDI file.

    If a sample size is given, then only parse that number of files.

    Parameters:
        genre (str): The genre of music.
        sample_size (int): The number of MIDI files to parse.

    Returns:
        pieces (list[list[Symbol]]): The symbols of each MIDI file.
    '''
    midi_fp = f'resources/{genre}'

    midi_files = [file for file in Path(midi_fp).rglob('*.mid')]
    shuffle(midi_files)
    if sample_size:
        midi_files = midi_files[:sample_size]
    
    pieces = thread_map(_get_symbols, midi_files, max_workers=6)

    return pieces