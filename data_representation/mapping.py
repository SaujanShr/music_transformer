from data_representation.common import (
    Token,
    PITCHES, TIME_SHIFTS,
    NoteOn, NoteOff,
    TimeShift, Instrument,
    Start, End
)

class Mapping:
    def _increment(self) -> int:
        self.counter += 1
        return self.counter

    def __init__(self):
        self.counter = -1

        self.start_mapping = self._increment()
        self.end_mapping = self._increment()
        
        self.note_on_mapping = {
            pitch:self._increment()
            for pitch in PITCHES
        }
        self.note_off_mapping = {
            pitch:self._increment()
            for pitch in PITCHES
        }
        self.time_shift_mapping = {
            cs:self._increment()
            for cs in TIME_SHIFTS
        }
        self.instrument_mapping = {}

        self.mapping_dict = {
            self.start_mapping:Start(),
            self.end_mapping:End()
        } | {
            mapping:NoteOn(pitch)
            for (pitch, mapping) in self.note_on_mapping.items()
        } | {
            mapping:NoteOff(pitch)
            for (pitch, mapping) in self.note_off_mapping.items()
        } | {
            mapping:TimeShift(cs)
            for (cs, mapping) in self.time_shift_mapping.items()
        }

    def token_to_mapping(self, token:Token) -> int:
        if isinstance(token, NoteOn): return self.note_on_mapping[token.pitch]
        elif isinstance(token, NoteOff): return self.note_off_mapping[token.pitch]
        elif isinstance(token, TimeShift): return self.time_shift_mapping[token.cs]
        elif isinstance(token, Instrument): return self.instrument_mapping[token.instrument]
        elif isinstance(token, Start): return self.start_mapping
        elif isinstance(token, End): return self.end_mapping
        else: raise Exception(f'Unknown token:{token}')

    def tokens_to_mapping(self, tokens:list[Token]):
        return [
            self.token_to_mapping(token)
            for token in tokens
        ]

    def mapping_to_token(self, mapping:int) -> Token:
        return self.mapping_dict[mapping]

    def size(self) -> int:
        return len(self.mapping_dict)

    def add_instruments(self, instruments:set[str]):
        new_instruments = [
            (instrument,self._increment())
            for instrument in instruments
        ]

        self.instrument_mapping |= {
            instrument:mapping
            for (instrument, mapping) in new_instruments
        }
        self.mapping_dict |= {
            mapping:Instrument(instrument)
            for (instrument, mapping) in new_instruments
        }