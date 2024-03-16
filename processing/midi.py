from music21.stream import Stream, Part
from music21.instrument import fromString
from music21.note import Note, Unpitched
from music21.pitch import Pitch
from music21.duration import Duration
from music21.tempo import MetronomeMark

def _get_instrument(instrument):
    return fromString(instrument)


def _partition_by_instrument(symbols):
    partitions = {}

    for symbol in symbols:
        instrument = symbol.instrument

        if instrument not in partitions:
            partitions[instrument] = []
        partitions[instrument].append(symbol)

    return [
        (_get_instrument(instrument), part)
        for (instrument, part) in partitions.items()
    ]


def midi(symbols, file_path):
    parts = _partition_by_instrument(symbols)

    stream = Stream()

    for (instrument, symbols) in parts:
        part = Part()
        stream.append(part)

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

    stream.write('midi', fp=file_path)