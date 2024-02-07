from dataclasses import dataclass, field

from music21.key import Key
from music21.instrument import Instrument
from music21.note import Note, Rest
from music21.chord import Chord

@dataclass
class Track:
    key: str = 'C:major'
    instrument: str = None
    notes: list[list[str]] = field(default_factory=list) 
    durations: list[float] = field(default_factory=list)
    offsets: list[float] = field(default_factory=list)

    def set_instrument(self, instrument: Instrument):
        self.instrument = instrument.instrumentName.lower()

    def set_key(self, key: Key):
        self.key = f'{key.tonic.name}:{key.mode}'

    def add_chord(self, chord: Chord):
        names = [pitch.nameWithOctave for pitch in chord.pitches]
        duration = chord.duration.quarterLength
        offset = chord.offset

        self.notes.append(names)
        self.durations.append(duration)
        self.offsets.append(offset)

    def add_note(self, note: Note):
        names = [note.nameWithOctave]
        duration = note.duration.quarterLength
        offset = note.offset

        self.notes.append(names)
        self.durations.append(duration)
        self.offsets.append(offset)

    def add_rest(self, rest: Rest):
        names = [rest.name]
        duration = rest.duration.quarterLength
        offset = rest.offset

        self.notes.append(names)
        self.durations.append(duration)
        self.offsets.append(offset)
