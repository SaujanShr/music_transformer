from math import ceil

NOTE_VALUES = list(range(-1, 129))
TIME_SHIFT_VALUES = [round(i*0.01, 2) for i in range(1, 101)]

def _note_to_index(note: str) -> int:
    match note[1]:
        case '#': 
            return _note_to_index(note[0] + note[2:]) + 1
        case '-': 
            return _note_to_index(note[0] + note[2:]) - 1
        case n if n.isdecimal():
            key = ord(note[0]) - 65
            octave = int(note[1:]) * 12
            return key + octave
        case _:
            return -1   


class NoteOn:
    def tokenise(note: str) -> str:
        index = _note_to_index(note)

        if index < NOTE_VALUES[0] or index > NOTE_VALUES[-1]:
            index = NOTE_VALUES[0]

        return NoteOn.token(index)

    def token(index: int):
        return f'NoteOn:{index}'


class NoteOff:
    def tokenise(note: str) -> str:
        index = _note_to_index(note)

        if index < NOTE_VALUES[0] or index > NOTE_VALUES[-1]:
            index = NOTE_VALUES[0]

        return NoteOff.token(index)

    def token(index: int):
        return f'NoteOff:{index}'


class TimeShift:
    def tokenise(duration: float) -> str:
        cs = ceil(duration * 100) / 100

        if cs < TIME_SHIFT_VALUES[0] or cs > TIME_SHIFT_VALUES[-1]:
            cs = TIME_SHIFT_VALUES[0]

        return TimeShift.token(cs)

    def token(cs: int):
        return f'TimeShift:{cs}'


class Instrument:
    def tokenise(instrument: str) -> str:
        instrument = instrument.lower()
        
        return self.token(instrument)

    def token(instrument: str):
        return f'Instrument:{instrument}'


class Intermediate:
    def token() -> str:
        return 'Intermediate'

class Start:
    def token() -> str:
        return 'Start'

class End:
    def token() -> str:
        return 'End'