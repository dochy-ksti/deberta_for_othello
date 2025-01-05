import logging

from .arena import Arena

from .net_wrapper import NNetWrapper

from .mcts_args import MctsArgs

import sys

from .intf_py_communicator import PyCommunicator


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

log = logging.getLogger(__name__)


def compare_in_arena(dir: str):
    args = MctsArgs()
    try:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    except IndexError:
        log.info('you need two files to compare')
        return

    try:
        mode = sys.argv[3]
    except IndexError:
        mode = "cc"

    pc = PyCommunicator(args.is_release)
    if mode == "cc":
        net1 = NNetWrapper(pc, args, False)
        net2 = NNetWrapper(pc, args, False)
    elif mode == "dd":
        net1 = NNetWrapper(pc, args, True)
        net2 = NNetWrapper(pc, args, True)
    elif mode == "cd":
        net1 = NNetWrapper(pc, args, False)
        net2 = NNetWrapper(pc, args, True)
    else:
        raise ValueError(
            f"mode must be (cc/cd/dd). mode {mode} is not supported")

    log.info('loading files...')
    if file1 != "none":
        net1.load_checkpoint(folder=dir, filename=file1)
    if file2 != "none":
        net2.load_checkpoint(folder=dir, filename=file2)

    arena = Arena()

    p1wins, p2wins, draws = arena.play_games(pc, net1, net2)

    log.info("P1/P2 WINS : %d / %d ; DRAWS : %d" %
             (p1wins, p2wins, draws))


if __name__ == "__main__":
    compare_in_arena("target")
