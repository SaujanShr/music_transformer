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
        '''
        Initialise the lookup for token mappings.
        '''
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
        '''
        Increment and return the token ID counter.

        Returns:
            counter (int): The token ID counter.
        '''
        self.counter += 1
        return self.counter

    def update_instruments(self, tokens):
        '''
        Add in any new instruments found in the token sequence to the mapping.

        Parameters:
            tokens (list[Token]): The tokens.
        '''
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
        '''
        Get the integer token value of a token object.

        Parameters:
            token (Token): The token object.

        Returns:
            mapping (int): The integer token value.
        '''
        mapping = self.mapping[token.__str__()]

        return mapping

    def tokens_to_mapping(self, tokens):
        '''
        Get integer tokens from token objects.

        Parameters:
            tokens (list[Token]): The token objects.

        Returns:
            mapping (list[int]): The integer tokens.
        '''
        mapping = [
            self.token_to_mapping(token)
            for token in tokens
        ]
        return mapping

    def mapping_to_tokens(self, mapping):
        '''
        Get token objects from integer tokens.

        Parameters:
            mapping (list[int]): The integer tokens.

        Returns:
            tokens (list[Token]): The token objects.
        '''
        tokens = [
            self.mapping[i]
            for i in mapping
        ]

        return tokens
    
    def size(self):
        '''
        Get the vocabulary size.

        Returns:
            size (int): The vocabulary size.
        '''
        size = self.counter + 1

        return size
