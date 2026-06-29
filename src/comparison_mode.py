"""Compare viewer state machine (H3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set


class CompareEntry(Enum):
    BURST_GROUP = "burst_group"
    GALLERY_MULTISELECT = "gallery_multiselect"


@dataclass
class ComparisonSession:
    """Dual-pane compare session: fixed Select (left) / Candidate (right)."""

    members: List[str]
    entry: CompareEntry
    select: str = ""
    candidate: str = ""
    queue: List[str] = field(default_factory=list)
    rejected: Set[str] = field(default_factory=set)
    complete: bool = False

    def visible_members(self) -> List[str]:
        return [path for path in self.members if path not in self.rejected]

    def restart_round(self) -> None:
        visible = self.visible_members()
        if len(visible) >= 2:
            self.select = visible[0]
            self.candidate = visible[1]
            self.queue = visible[2:]
            self.complete = False
        else:
            self.select = visible[0] if visible else ""
            self.candidate = ""
            self.queue = []
            self.complete = True

    def _others(self) -> List[str]:
        return [path for path in self.visible_members() if path != self.select]

    def step_candidate(self, delta: int) -> None:
        others = self._others()
        if len(others) <= 1:
            self._handle_end_of_round()
            return
        try:
            index = others.index(self.candidate)
        except ValueError:
            self.candidate = others[0]
            return
        if delta > 0:
            if index >= len(others) - 1:
                self._handle_end_of_round()
                return
            self.candidate = others[index + 1]
        elif index <= 0:
            self.candidate = others[-1]
        else:
            self.candidate = others[index - 1]

    def promote_candidate(self) -> str:
        """Promote Candidate to Select; returns new Select path."""
        self.select = self.candidate
        self._replenish_candidate()
        return self.select

    def reject_candidate(self) -> None:
        if not self.candidate:
            return
        self.rejected.add(self.candidate)
        self._replenish_candidate()

    def reject_select(self) -> None:
        if not self.select:
            return
        self.rejected.add(self.select)
        self._replenish_select()

    def _pop_queue(self) -> str:
        while self.queue:
            path = self.queue.pop(0)
            if path not in self.rejected:
                return path
        return ""

    def _replenish_candidate(self) -> None:
        visible = self.visible_members()
        if len(visible) < 2:
            self.candidate = ""
            self.complete = True
            return
        next_path = self._pop_queue()
        if next_path and next_path != self.select:
            self.candidate = next_path
            return
        others = self._others()
        self.candidate = others[0] if others else ""
        if len(visible) < 2:
            self.complete = True

    def _replenish_select(self) -> None:
        visible = self.visible_members()
        if not visible:
            self.select = ""
            self.candidate = ""
            self.complete = True
            return
        next_path = self._pop_queue()
        if next_path:
            self.select = next_path
        else:
            self.select = visible[0]
        if self.candidate in self.rejected or self.candidate == self.select:
            self._replenish_candidate()
        elif len(visible) < 2:
            self.complete = True

    def _handle_end_of_round(self) -> None:
        visible = self.visible_members()
        if len(visible) >= 2:
            self.restart_round()
        else:
            self.complete = True
