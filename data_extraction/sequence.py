from math import floor

from data_extraction.piece import Piece
from data_extraction.track import Track
from data_extraction.token import NoteOn, NoteOff, TimeShift, Instrument, Intermediate, Start, End


def encode_track(track: Track) -> list[(float, list[str])]:
    encoded_track = {}

    for notes, duration, on_offset in zip(track.notes, track.durations, track.offsets):
        if not on_offset in encoded_track:
            encoded_track[on_offset] = []

        off_offset = on_offset + duration
        if not off_offset in encoded_track:
            encoded_track[off_offset] = []

        for note in notes:
            encoded_track[on_offset].append(NoteOn.tokenise(note))
            encoded_track[off_offset].append(NoteOff.tokenise(note))

    return sorted(encoded_track.items())


def time_shift_sequence(prev_offset: float, next_offset: float) -> list[float]:
    shift_duration = next_offset - prev_offset
    seconds = floor(shift_duration)
    
    time_shift_sequence = [TimeShift.tokenise(1) for _ in range(seconds)]

    if seconds != shift_duration:
        time_shift_sequence.append(TimeShift.tokenise(shift_duration % 1))

    return time_shift_sequence


def track_to_tokens(track: Track):
    encoded_track = encode_track(track)

    prev_offset = encoded_track[0][0]
    tokens = list(encoded_track[0][1])

    for (offset, notes) in encoded_track[1:]:
        tokens += time_shift_sequence(prev_offset, offset)
        tokens += notes

        prev_offset = offset
        
    return [f'Instrument<{track.instrument}>'] + tokens


def piece_to_token(piece: Piece) -> list[str]:
    track_token_sequences = [track_to_tokens(track) for track in piece.tracks]

    tokens = track_token_sequences[0]

    for track_tokens in track_token_sequences[1:]:
        tokens.append(Intermediate.token())
        tokens += track_tokens

    return [Start.token()] + tokens + [End.token()]


def pieces_to_tokens(pieces: list[Piece]) -> list[list[str]]:
    return [piece_to_token(piece) for piece in pieces]
