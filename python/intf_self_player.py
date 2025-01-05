from ctypes import c_void_p, c_size_t, POINTER, CDLL
from numpy.typing import NDArray
from numpy import float32

from .intf_carray import CArray
from .train_example import TrainExample


class SelfPlayer:
    def __init__(self, lib: CDLL, p: c_void_p):
        self.lib = lib
        self.p = p

    def __del__(self):
        self.lib.destroy_self_player(self.p)

    #  NスレッドでN個の試合を同時にシミュレーションしている。一手進めて盤面を返す
    #
    #  最初の指し手は必ずplayer1とする。
    #
    #  player: isize
    #  0ならplayerを問わない
    #  1ならplayer1の盤面を準備する
    #  -1ならplayer2の盤面を準備する
    #
    #  mixed_thinker: bool
    #  trueの場合、バッチの後半半分のthinkerを逆にする。これによりArenaで1バッチで先手後手を半々に分けることが可能
    #
    #  戻り値:
    #  0: まだ準備が出来ていない
    #  1: 盤面の準備が出来た
    #  2: すべての試合が既に終わっていて、トレーニング用のデータの準備が出来た
    def prepare_next(self, player: int, mixed_thinker: bool) -> int:
        return self.lib.self_player_prepare_next(self.p, player, mixed_thinker)

    # BATCH_SIZE * BOARD_SIZE
    def get_boards_for_prediction(self, player: int, mixed_thinker: bool) -> NDArray[float32]:
        carray = self.lib.self_player_get_boards_for_prediction(self.p, player, mixed_thinker)
        return CArray(self.lib, carray).to_numpy()

    # BATCH_SIZE * MOVE_LEN
    def get_pis_for_training(self) -> NDArray[float32]:
        return CArray(self.lib, self.lib.self_player_get_pis_for_training(self.p)).to_numpy()

    # BATCH_SIZE * BOARD_SIZE
    def get_boards_for_training(self) -> NDArray[float32]:
        return CArray(self.lib, self.lib.self_player_get_boards_for_training(self.p)).to_numpy()

    # BATCH_SIZE
    def get_players_for_training(self) -> NDArray[float32]:
        return CArray(self.lib, self.lib.self_player_get_players_for_training(self.p)).to_numpy()

    # BATCH_SIZE
    def get_results_for_training(self) -> NDArray[float32]:
        return CArray(self.lib, self.lib.self_player_get_results_for_training(self.p)).to_numpy()
    
    def get_results_for_counting(self) -> NDArray[float32]:
        return CArray(self.lib, self.lib.self_player_get_results_for_counting(self.p)).to_numpy()

    def receive_prediction(self, pis: NDArray[float32], win_rates: NDArray[float32], player: int, mixed_thinker: bool):
        c_pis = CArray.from_numpy(self.lib, pis)
        c_win_rates = CArray.from_numpy(self.lib, win_rates)
        self.lib.self_player_receive_prediction(
            self.p, c_pis.p, c_win_rates.p, player, mixed_thinker)

    def get_train_examples(self) -> list[TrainExample]:
        pis = self.get_pis_for_training()
        boards = self.get_boards_for_training()
        players = self.get_players_for_training()
        results = self.get_results_for_training()
        return [TrainExample(board, player, pi, result) for pi, board, player, result in zip(pis, boards, players, results)]


def define_self_player_funcs(lib: CDLL):
    lib.create_self_player.argtypes = [POINTER(c_void_p), c_size_t]
    lib.create_self_player.restype = POINTER(c_void_p)
    lib.destroy_self_player.argtypes = [POINTER(c_void_p)]

    lib.self_player_prepare_next.argtypes = [
        POINTER(c_void_p), c_size_t, c_size_t]
    lib.self_player_prepare_next.restype = c_size_t
    lib.self_player_get_boards_for_prediction.argtypes = [
        POINTER(c_void_p), c_size_t, c_size_t]
    lib.self_player_get_boards_for_prediction.restype = POINTER(
        c_void_p)
    lib.self_player_get_pis_for_training.argtypes = [
        POINTER(c_void_p)]
    lib.self_player_get_pis_for_training.restype = POINTER(
        c_void_p)
    lib.self_player_get_boards_for_training.argtypes = [
        POINTER(c_void_p)]
    lib.self_player_get_boards_for_training.restype = POINTER(
        c_void_p)
    lib.self_player_get_players_for_training.argtypes = [
        POINTER(c_void_p)]
    lib.self_player_get_players_for_training.restype = POINTER(
        c_void_p)
    lib.self_player_get_results_for_training.argtypes = [
        POINTER(c_void_p)]
    lib.self_player_get_results_for_training.restype = POINTER(
        c_void_p)
    lib.self_player_get_results_for_counting.argtypes = [
        POINTER(c_void_p)]
    lib.self_player_get_results_for_counting.restype = POINTER(
        c_void_p)
    lib.self_player_receive_prediction.argtypes = [
        POINTER(c_void_p), POINTER(c_void_p), POINTER(c_void_p), c_size_t, c_size_t]
