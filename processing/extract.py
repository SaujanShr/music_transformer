from pathlib import Path

from tqdm.contrib.concurrent import process_map
from itertools import islice, chain

from numpy import array, load, savez

from music21.converter import parse
from music21.stream.base import Stream
from music21.instrument import Instrument, partitionByInstrument
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
        part (Stream): Stream partition

    Returns:
        symbols (list[Symbol]): List of symbols
    '''
    if part.getInstrument() == None:
        instrument = Instrument()
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
    Parse the midi file to a stream and extract symbols from the stream.

    Parameters:
        file (Path): Path to midi file

    Returns:
        symbols (list[Symbol]): List of symbols
    '''
    midi = parse(file)
    parts = partitionByInstrument(midi)

    part_symbols = [_part_to_symbols(part) for part in parts]
    symbols = list(chain.from_iterable(part_symbols))
    symbols = sorted(symbols, key=lambda symbol:symbol.offset)

    return symbols


def extract(genre, sample_size=None):
    '''
    Parse the midi files from the given genre of music
    and extract symbols for each midi file.

    If a sample size is given, then only parse that number of files.

    Parameters:
        genre (str): Genre of music
        sample_size (int): Number of midi files to parse

    Returns:
        pieces (list[list[Symbol]]): List of symbols for each midi file
    '''
    dataset_fp = f'resources/{genre}_{sample_size}.npz'
    midi_fp = f'resources/{genre}'

    # If the midi files have already been parsed and saved, load the save.
    if Path(dataset_fp).is_file():
        data = load(dataset_fp, allow_pickle=True)
        return data['data'].tolist()


    midi_files = Path(midi_fp).rglob('*.mid')
    if sample_size:
        midi_files = islice(midi_files, sample_size)

    pieces = process_map(_get_symbols, midi_files, max_workers=6)
    data = array(pieces, dtype=object)

    # Save the parsed files to load next time.
    savez(dataset_fp, data=data)

    return pieces