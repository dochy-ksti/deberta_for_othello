import logging

from .coach import Coach
from .mcts_args import MctsArgs

import sys

from .intf_py_communicator import PyCommunicator


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

log = logging.getLogger(__name__)


def main():
    args = None
    try:
        if sys.argv[1] == "release":
            args = MctsArgs()
            args.is_release = True
        else:
            raise ValueError(f"{sys.argv[1]} release のみ有効")

    except IndexError:
        args = MctsArgs()

    # if args.load_model:
    #     log.info('Loading checkpoint "%s/%s"...',
    #              args.load_folder_file[0], args.load_folder_file[1])
    #     nnet.load_checkpoint(
    #         args.load_folder_file[0], args.load_folder_file[1])
    # else:
    #     log.warning('Not loading a checkpoint!')

    # if args.load_model:
    #     log.info("Loading 'train_examples' from file...")
    #     c.load_train_examples()

    pc = PyCommunicator(args.is_release)
    c = Coach(pc, args)

    log.info('Starting the learning process')
    c.learn()




if __name__ == "__main__":
    main()
