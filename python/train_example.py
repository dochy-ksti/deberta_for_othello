from dataclasses import dataclass
from typing import Sequence
from numpy.typing import NDArray
from numpy import float32

import numpy as np

@dataclass
class TrainExample:
    canonical_board: NDArray[float32]
    cur_player: int
    pi: NDArray[float32]
    v: int

    def to_str(self, title: str) -> str:
        return '\n'.join([title, board_to_str(self.canonical_board * self.cur_player),
                          f"player {self.cur_player} result {self.v} diff {board_to_diff(self.canonical_board * self.cur_player)}",
                          pi_to_str(self.pi)])


def train_examples_to_str(l: Sequence[TrainExample]) -> str:
    return "\n".join([ex.to_str(f"ex {index}") for index, ex in enumerate(l)])


def pi_to_str(pi: NDArray[float32]) -> str:
    return " ".join([f"{p:.2f}" for p in pi])


def board_to_str(board: NDArray[float32]) -> str:
    return '\n'.join(["".join(
        ['🔴' if c == 0 else '⚫️' if c == 1 else '⚪️' if c == -1 else "Unsupported" for c in line])
        for line in board])


def board_to_diff(board: NDArray[float32]) -> int:
    return np.count_nonzero(board == 1) - np.count_nonzero(board == -1)
