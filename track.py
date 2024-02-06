from music21.key import Key
from music21.instrument import Instrument
from music21.meter import TimeSignature
from music21.note import Note
from music21.chord import Chord

class Track:
    def __init__(self):
        self.key = 'C:major'
        self.instrument = None
        self.time_signature = None

        self.notes = []
        self.durations = []

    def set_instrument(self, instrument: Instrument):
        self.instrument = instrument.instrumentName

    def set_key(self, key: Key):
        self.key = f'{key.tonic.name}:{key.mode}'

    def set_time_signature(self, time_signature: TimeSignature):
        self.time_signature = time_signature

    def add_chord(self, chord: Chord):
        names = [pitch.nameWithOctave for pitch in chord.pitches]
        duration = str(chord.duration.quarterLength)

        self.notes.append(names)
        self.durations.append(duration)

    def add_note(self, note: Note):
        names = [note.nameWithOctave]
        duration = str(note.duration.quarterLength)

        self.notes.append(names)
        self.durations.append(duration)

    def add_rest(self, rest: Note):
        names = [rest.name]
        duration = str(rest.duration.quarterLength)

        self.notes.append(names)
        self.durations.append(duration)