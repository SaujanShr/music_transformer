from numpy import save, load

from common.common import (
    PITCHES, TIME_SHIFTS, TEMPOS
)
from common.token import (
    NoteOn, NoteOff, Tempo,
    TimeShift, Instrument,
    Start, End
)

class Lookup:
    def __init__(self):
        self.counter = -1

        self.mapping = {}

        start_mapping = { self._increment() : Start() }
        end_mapping = { self._increment() : End() }

        note_on_mapping = {
            self._increment() : NoteOn(pitch)
            for pitch in PITCHES
        }
        note_off_mapping = {
            self._increment() : NoteOff(pitch)
            for pitch in PITCHES
        }
        time_shift_mapping = {
            self._increment() : TimeShift(cs)
            for cs in TIME_SHIFTS
        }
        tempo_mapping = {
            self._increment() : Tempo(tempo)
            for tempo in TEMPOS
        }

        self.mapping = start_mapping | end_mapping | \
            note_on_mapping | note_off_mapping | \
            time_shift_mapping | tempo_mapping

        self.mapping |= {
            token.__str__() : mapping
            for (mapping, token) in self.mapping.items()
        }


    def _increment(self):
        self.counter += 1
        return self.counter

    def update_instruments(self, tokens):
        instruments = { 
            token.instrument for token in tokens 
            if isinstance(token, Instrument)
        }
        unknown_mapping = {
            self._increment() : Instrument(instrument)
            for instrument in instruments
            if not Instrument(instrument).__str__() in self.mapping
        }

        self.mapping |= unknown_mapping
        self.mapping |= {
            token.__str__() : mapping
            for (mapping, token) in unknown_mapping.items()
        }

    def token_to_mapping(self, token):
        return self.mapping[token.__str__()]

    def tokens_to_mapping(self, tokens):
        mapping = [
            self.token_to_mapping(token)
            for token in tokens
        ]
        return mapping

    def mapping_to_tokens(self, mapping):
        return [
            self.mapping[i]
            for i in mapping
        ]
    
    def size(self):
        return self.counter + 1
