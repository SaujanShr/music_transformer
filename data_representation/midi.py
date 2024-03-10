from music21.stream import Stream, Part
from music21.note import Note, Unpitched
from music21.pitch import Pitch
from music21.duration import Duration
from music21.instrument import Instrument, UnpitchedPercussion, fromString
from music21.tempo import MetronomeMark

from data_representation.common import (
    Symbol, Piece
)

import re
 
def camel_case_split(instrument):
    split = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', instrument)
    return ' '.join(split)

def get_instrument(instrument:str|None) -> Instrument:
    if instrument == None:
        return UnpitchedPercussion()
    else:
        cleaned_instrument = camel_case_split(instrument)
        return fromString(cleaned_instrument)


def partition_by_instrument(piece: Piece) -> list[(Instrument, Piece)]:
    partitions = {}

    for symbol in piece:
        if symbol.pitch == -1:
            if None not in partitions:
                partitions[None] = []
            partitions[None].append(symbol)
        else:
            instrument = symbol.instrument
            if instrument not in partitions:
                partitions[instrument] = []
            partitions[instrument].append(symbol)
    
    return [
        (get_instrument(instrument), part) 
        for (instrument, part) in partitions.items()
    ]

def piece_to_midi(piece: Piece):
    piece_parts = partition_by_instrument(piece)

    stream = Stream()

    for (instrument, piece_part) in piece_parts:
        part = Part()
        stream.append(part)

        part.append(instrument)
        part.append(MetronomeMark(218))

        for symbol in piece_part:
            if symbol.pitch == -1:
                instrument = get_instrument(symbol.instrument)

                note = Unpitched()
                note.storedInstrument = instrument
            else:
                pitch = Pitch(midi=symbol.pitch)

                note = Note()
                note.pitch = pitch

            part.append(note)

            note.duration = Duration(symbol.duration)
            note.offset = symbol.offset


    fp = stream.write('midi', fp='test.midi')