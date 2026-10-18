"""Who is using BeamNG right now: the Tracks tab, a training run, or a model replay.

Only one of them may drive the game at a time. The Tracks tab holds the game
softly (it hands it over on request via its release callback); training and
replays hold it until their process exits.
"""

from __future__ import annotations

from typing import Callable


class GameLock:
    def __init__(self, on_change: Callable[[], None] | None = None):
        self.owner: str | None = None
        self._release: Callable[[], None] | None = None
        self._on_change = on_change

    def claim(self, owner: str, release: Callable[[], None] | None = None) -> bool:
        """Take the game for `owner`. A soft holder (one with a release callback) is asked to let go."""
        if self.owner not in (None, owner):
            if self._release is None:
                return False  # a hard holder (training / replay) is still running
            release_fn, self._release = self._release, None
            release_fn()
        self.owner, self._release = owner, release
        self._changed()
        return True

    def release(self, owner: str) -> None:
        if self.owner == owner:
            self.owner, self._release = None, None
            self._changed()

    def _changed(self) -> None:
        if self._on_change:
            self._on_change()
