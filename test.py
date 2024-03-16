from music21 import converter, instrument, tempo
from music21 import note, percussion, key, chord, stream
from pathlib import Path

midi_files = Path('resources/test').rglob('*.mid')

stream = stream.Stream()

for file in midi_files:
    midi = converter.parse(file)
    parts = instrument.partitionByInstrument(midi)[7]
    # for element in midi.flatten():
    #     if isinstance(element, note.Unpitched):
    #         # print(element.storedInstrument())
    #         # stream.append(element)
    #         pass
    #     if isinstance(element, percussion.PercussionChord):
    #         stream.append(element)
    stream.append(parts)
    print("end")

stream.show('text')
fp = stream.write('midi', fp='test.midi')