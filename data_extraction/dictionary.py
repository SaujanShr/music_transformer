from data_extraction.token import (
    NOTE_VALUES, TIME_SHIFT_VALUES, 
    NoteOn, NoteOff, TimeShift, 
    Instrument, Intermediate, Start, End
)

class Dictionary:
    def _increment(self) -> int:
        self.counter += 1
        return self.counter
    
    def __init__(self):
        self.counter = -1

        self.mapping_dictionary = {
            NoteOn.token(index):self._increment() 
            for index in NOTE_VALUES
        } | {
            NoteOff.token(index):self._increment()
            for index in NOTE_VALUES
        } | {
            TimeShift.token(cs):self._increment()
            for cs in TIME_SHIFT_VALUES
        } | {
            Intermediate.token():self._increment(),
            Start.token():self._increment(),
            End.token():self._increment()
        }

        self.token_dictionary = {
            mapping:token
            for (token, mapping) in self.mapping_dictionary.items()
        }

    def add_instruments(self, instruments: list[str]):
        instrument_mapping = [
            (Instrument.token(instrument), self._increment())
            for instrument in set(instruments)
        ]
        self.mapping_dictionary |= {
            token:mapping
            for (token, mapping) in instrument_mapping
        }
        self.token_dictionary |= {
            mapping:token
            for (token, mapping) in instrument_mapping
        }

    def mapping(self, token: str) -> int:
        return self.mapping_dictionary[token]

    def map_tokens(self, tokens: list[str]) -> list[int]:
        return [
            self.mapping_dictionary[token] 
            for token in tokens
        ]

    def token(self, mapping: int) -> str:
        return self.token_dictionary[mapping]

    def tokenise_mappings(self, mappings: list[int]) -> list[str]:
        return [
            self.token_dictionary[mapping] 
            for mapping in mappings
        ]

    def size(self):
        return len(self.mapping_dictionary)