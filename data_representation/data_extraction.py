from pathlib import Path
from tqdm.contrib.concurrent import process_map
from itertools import chain, islice
import numpy as np
from music21.converter import parse
from music21.instrument import Instrument, partitionByInstrument
from music21.chord import Chord
from music21.percussion import PercussionChord
from music21.note import Note, Unpitched
from music21.stream.base import Stream

from data_representation.common import Symbol, Piece


def part_to_symbols(part:Stream) -> list[Symbol]:
    if part.getInstrument() == None:
        return []
    
    instrument = part.getInstrument()
    symbols = []

    for element in part.flatten():
        if isinstance(element, Chord):
            symbols += [
                Symbol.get(instrument, note.pitch, note.duration, element.offset + note.offset)
                for note in element.notes
            ]
        elif isinstance(element, PercussionChord):
            symbols += [
                Symbol.get(note.storedInstrument, None, note.duration, element.offset + note.offset)
                for note in element.notes
            ]
        elif isinstance(element, Note):
            symbols.append(
                Symbol.get(instrument, element.pitch, element.duration, element.offset)
            )
        elif isinstance(element, Unpitched):
            symbols.append(
                Symbol.get(element.storedInstrument, None, element.duration, element.offset)
            )
        
    return symbols


def get_piece(path:Path) -> Piece:
    midi = parse(path)
    parts = partitionByInstrument(midi)

    symbols = list(chain(*[part_to_symbols(part) for part in parts]))
    piece = sorted(symbols, key=lambda symbol:symbol.offset)

    return piece


def get_pieces(midi_files:list[Path], sample_size:int) -> list[Piece]:
    pieces = process_map(
        get_piece,
        islice(midi_files, sample_size),
        max_workers=6
    )

    return np.array(pieces, dtype=object)


def get_pieces_from_genre(genre:str, sample_size:int=10) -> list[Piece]:
    if Path(f'resources/{genre}.npz').is_file():
        data = np.load(f'resources/{genre}.npz', allow_pickle=True)
        return data['data']

    midi_files = Path(f'resources/{genre}').rglob('*.mid')
    pieces = get_pieces(midi_files, sample_size)
    
    np.savez(f'resources/{genre}.npz', data=pieces)

    return pieces


def get_instruments(pieces:list[Piece]) -> set[str]:
    return set.union(*[
        { symbol.instrument for symbol in piece } 
        for piece in pieces
    ])
