from ctypes import c_void_p, c_size_t, POINTER, CDLL
import ctypes
import numpy as np

from .intf_self_player import SelfPlayer, define_self_player_funcs
from .intf_carray import define_carray_funcs


class PyCommunicator:
    def __init__(self, is_release: bool):
        if is_release:
            self.lib = ctypes.cdll.LoadLibrary(
                'target/release/rust_othello_alphazero.dll')
        else:
            self.lib = ctypes.cdll.LoadLibrary(
            'target/debug/rust_othello_alphazero.dll')
        define_py_communicator_funcs(self.lib)
        define_self_player_funcs(self.lib)
        define_carray_funcs(self.lib)
        self.p = self.lib.create_py_communicator()

    def __del__(self):
        self.lib.destroy_py_communicator(self.p)

    def create_self_player(self, player_mode: int) -> SelfPlayer:
        return SelfPlayer(self.lib, self.lib.create_self_player(self.p, player_mode))

    def size_y(self) -> int:
        return self.lib.size_y()

    def size_x(self) -> int:
        return self.lib.size_x()

    def batch_size(self) -> int:
        return self.lib.batch_size()

    def move_len(self) -> int:
        return self.lib.move_len()

    def board_size(self) -> int:
        return self.lib.board_size()


def define_py_communicator_funcs(lib: CDLL):
    lib.create_py_communicator.restype = POINTER(c_void_p)
    lib.destroy_py_communicator.argtypes = [POINTER(c_void_p)]
    lib.batch_size.restype = c_size_t
    lib.size_x.restype = c_size_t
    lib.size_y.restype = c_size_t
    lib.move_len.restype = c_size_t
    lib.board_size.restype = c_size_t


if __name__ == "__main__":
    lib = ctypes.cdll.LoadLibrary(
        '../../target/debug/rust_othello_alphazero.dll')
    pc = lib.create_py_communicator()
    c_array = lib.create_c_array_from_communicator(pc)
    # value = lib.c_array_get3(c_array, 0, 0, 0)
    # print(value)
    c_array_ptr = lib.c_array_as_ptr(c_array)
    np_array = np.ctypeslib.as_array(
        c_array_ptr, (lib.batch_size(), lib.size1(), lib.size2())).copy()

    lib.destroy_c_array(c_array)
    lib.destroy_py_communicator(pc)
    print(np_array)
    print('done')
