"""
Bitmask representation of team availability (spec section 3).

A TeamRegistry assigns each team a stable bit index so that "my used teams"
and "an opponent's used teams" can both be represented as plain Python ints
and combined/queried with fast bitwise ops instead of set operations.
"""
from __future__ import annotations


class TeamRegistry:
    def __init__(self, teams: list[str] | None = None):
        self._team_to_bit: dict[str, int] = {}
        self._bit_to_team: dict[int, str] = {}
        if teams:
            for t in teams:
                self.register(t)

    def register(self, team: str) -> int:
        if team not in self._team_to_bit:
            idx = len(self._team_to_bit)
            bit = 1 << idx
            self._team_to_bit[team] = bit
            self._bit_to_team[bit] = team
        return self._team_to_bit[team]

    def bit(self, team: str) -> int:
        if team not in self._team_to_bit:
            self.register(team)
        return self._team_to_bit[team]

    def team(self, bit: int) -> str:
        return self._bit_to_team[bit]

    def mask_from_teams(self, teams: list[str]) -> int:
        mask = 0
        for t in teams:
            mask |= self.bit(t)
        return mask

    def teams_from_mask(self, mask: int) -> list[str]:
        out = []
        b = 1
        while mask:
            if mask & 1:
                out.append(self._bit_to_team.get(b, f"<unregistered:{b}>"))
            mask >>= 1
            b <<= 1
        return out

    def is_used(self, mask: int, team: str) -> bool:
        return bool(mask & self.bit(team))

    def __len__(self):
        return len(self._team_to_bit)
