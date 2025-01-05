import logging

from numpy.typing import NDArray
from numpy import float32

import numpy as np

from .intf_py_communicator import PyCommunicator

from .net_wrapper import NNetWrapper

from .intf_self_player import SelfPlayer

log = logging.getLogger(__name__)


class Arena:
    def __init__(self):
        pass

    def play_game(self, sp: SelfPlayer, net1: NNetWrapper, net2: NNetWrapper) -> NDArray[float32]:
        turn = 0
        cur_player = -1
        while True:
            turn += 1
            cur_player *= -1
            rnum = sp.prepare_next(cur_player, True)
            if rnum == 0:
                continue
            elif rnum == 1:
                boards = sp.get_boards_for_prediction(cur_player, True)
                if cur_player == 1:
                    pis, win_rates = net1.predict(boards)
                else:
                    pis, win_rates = net2.predict(boards)
                sp.receive_prediction(pis, win_rates, cur_player, True)
            elif rnum == 2:
                results = sp.get_results_for_counting()
                return results

    # def play_game_and_count_wons(self, sp: SelfPlayer, net1: NNetWrapper, net2: NNetWrapper) -> tuple[int, int, int]:
    #     array = self.play_game(sp, net1, net2)
    #     return (np.count_nonzero(array == 1.0), np.count_nonzero(array == -1.0), np.count_nonzero(array == 0.0))

    def play_games(self, pc: PyCommunicator, nnet: NNetWrapper, pnet: NNetWrapper) -> tuple[int, int, int]:
        array = self.play_game(pc.create_self_player(2), nnet, pnet)
        batch_size = pc.batch_size() // 2
        nwon1, pwon1, draws1 = (np.count_nonzero(array[:batch_size] == 1.0), np.count_nonzero(
            array[:batch_size] == -1.0), np.count_nonzero(array[:batch_size] == 0.0))
        pwon2, nwon2, draws2 = (np.count_nonzero(array[batch_size:] == 1.0), np.count_nonzero(
            array[batch_size:] == -1.0), np.count_nonzero(array[batch_size:] == 0.0))

        log.info(
            f"NNET WIN RATE FIRST {nwon1/(nwon1+pwon1+draws1)} SECOND {nwon2/(nwon2+pwon2+draws2)}")

        return nwon1 + nwon2, pwon1 + pwon2, draws1 + draws2
