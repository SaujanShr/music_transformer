from pathlib import Path
from music21 import converter, instrument, meter, chord, note, key
from itertools import islice
import numpy as np

from music21.stream.base import Part
from piece import Piece
from track import Track

def get_track(part: Part) -> Track:
    score = part.chordify().flatten()

    track = Track()

    for element in score:
        if isinstance(element, instrument.Instrument):
            track.set_instrument(element)
            continue

        match type(element):
            case meter.base.TimeSignature:
                track.set_time_signature(element)
            case chord.Chord:
                track.add_chord(element)
            case note.Note:
                track.add_note(element)
            case note.Rest:
                track.add_rest(element)
            case key.Key:
                track.set_key(element)
            # case _:
            #     print(f'skipped:{element}')

    return track

def get_piece(path: Path) -> Piece:
    midi = converter.parse(path)
    parts = instrument.partitionByInstrument(midi)

    piece = Piece()

    for part in parts:
        track = get_track(part)

        piece.add_track(track)

    return piece


def get_pieces(midi_files: Generator, sample_size:int=10) -> list[Pieces]:
    pieces = np.array([], dtype=object)

    for file in islice(midi_files, sample_size):
        piece = get_piece(file)
        pieces = np.append(pieces, piece)

    return pieces

def get_pieces_from_genre(genre: str) -> list[Piece]:
    if Path(f'datasets/{genre}.npz'.is_file()):
        data = np.load(savedata_path, allow_pickle=True)
        return data['pieces']

    midi_files = Path(f'datasets/{genre}').rglob('*.mid')
    pieces = get_pieces(midi_files)

    # np.savez(f'datasets/{genre}.npz', pieces=pieces)