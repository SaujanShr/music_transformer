from dataclasses import dataclass, field

from music21.stream.base import Part
from music21 import chord, note, key

from data_extraction.track import Track

@dataclass
class Piece:
    tracks: list[Track] = field(default_factory=list) 

    def add_track(self, track: Track) -> None:
        self.tracks.append(track)

    def add_part(self, part: Part) -> None:
        score = part.chordify().flatten()

        track = Track()

        for element in score:
            match type(element):
                case chord.Chord:
                    track.add_chord(element)
                case note.Note:
                    track.add_note(element)
                case note.Rest:
                    track.add_rest(element)
                case key.Key:
                    track.set_key(element)
                # case _:
                #     print(f'skipped:{element}')
        
        if track.notes:
            track.set_instrument(part.getInstrument())
            self.add_track(track)

    def get_instruments(self) -> list[str]:
        return [track.instrument for track in self.tracks]
