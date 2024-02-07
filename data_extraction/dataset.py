from pathlib import Path
from tqdm.contrib.concurrent import process_map
from music21 import converter, instrument
from itertools import islice
import numpy as np

from data_extraction.piece import Piece


def get_piece(path: Path) -> Piece:
    midi = converter.parse(path)
    parts = instrument.partitionByInstrument(midi)

    piece = Piece()

    for part in parts:
        piece.add_part(part)

    return piece


def get_pieces(midi_files, sample_size:int) -> list[Piece]:
    pieces = process_map(
        get_piece, 
        islice(midi_files, sample_size), 
        max_workers=6
    )

    return np.array(pieces, dtype=object)


def get_pieces_from_genre(genre: str, sample_size:int=10) -> list[Piece]:
    if Path(f'datasets/{genre}.npz').is_file():
        data = np.load(f'datasets/{genre}.npz', allow_pickle=True)
        return data['data']

    midi_files = Path(f'datasets/{genre}').rglob('*.mid')
    pieces = get_pieces(midi_files, sample_size)
    
    np.savez(f'datasets/{genre}.npz', data=pieces)

    return pieces
