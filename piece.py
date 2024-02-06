from track import Track

class Piece:
    def __init__(self):
        self.tracks = []

    def add_track(self, track: Track):
        self.tracks.append(track)