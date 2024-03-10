from music21 import converter, instrument, tempo
from music21 import note, percussion, key, chord
from pathlib import Path

midi_files = Path('resources/test').rglob('*.mid')

for file in midi_files:
    midi = converter.parse(file)
    parts = instrument.partitionByInstrument(midi)
    print("start")
    # print(parts[5].getInstrument().__class__.__name__)
    # midi.show('text')
    # print(parts[3].show('text'))
    for element in midi.flatten():
        if (isinstance(element, tempo.MetronomeMark)):
            print(element.number)
    print("end")

# from music21 import pitch

# p = pitch.Pitch(11)
# print(p.octave)