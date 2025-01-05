from collections import deque
import logging

from .prepare_training_data import choose_training_data


from .mcts_args import MctsArgs
from .intf_self_player import SelfPlayer
from .intf_py_communicator import PyCommunicator
from .train_example import TrainExample  # , train_examples_to_str
from .arena import Arena
from .net_wrapper import NNetWrapper as NNet

import numpy as np

log = logging.getLogger(__name__)


class Coach:

    def __init__(self, pc: PyCommunicator, args: MctsArgs):
        self.pc = pc
        self.nnet = NNet(pc, args, args.use_deberta)
        self.pnet = NNet(pc, args, args.use_deberta)  # the competitor network
        self.args = args
        self.cur_player = 1
        self.n = pc.size_x()
        self.train_examples_history: deque[list[TrainExample]] = deque()

    def learn(self):
        for i in range(1_000_000):
            log.info(f"iter {i+1}")
            train_examples = self.make_train_example(
                self.pc.create_self_player(1))

            log.info("Self Play Done")

            train_examples = [te for tes in train_examples for te in self.get_symmetries(
                 tes)]

            self.train_examples_history.append(train_examples)
            gathered, is_triangle = choose_training_data(
                self.train_examples_history, self.args)

            if is_triangle and (i - self.args.history_core_len) % 2 == 1:
                log.info(
                    f"Removing the oldest entry in trainExamples. len(examples) = {len(self.train_examples_history)}"
                )
                self.train_examples_history.popleft()
            else:
                log.info(
                    f"current history_len == {len(self.train_examples_history)}")

            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )

            if self.args.compare_with_self:
                self.pnet.load_checkpoint(
                    folder=self.args.checkpoint, filename="temp.pth.tar"
                )
            else:
                self.pnet = NNet(self.pc, self.args, self.args.use_deberta)

            self.nnet.train(gathered)

            if self.args.do_arena:
                log.info("PITTING AGAINST PREVIOUS VERSION")
                arena = Arena()

                nwins, pwins, draws = arena.play_games(
                    self.pc, self.nnet, self.pnet)

                log.info("NEW/PREV WINS : %d / %d ; DRAWS : %d" %
                         (nwins, pwins, draws))

                if (
                    pwins + nwins == 0
                    or float(nwins) / (pwins + nwins) < self.args.update_threshold
                ):
                    if self.args.enable_rejecting:
                        log.info("REJECTING NEW MODEL")
                        self.nnet.load_checkpoint(
                            folder=self.args.checkpoint, filename="temp.pth.tar"
                        )
                        self.train_examples_history.pop()
                    else:
                        log.info(
                            "The win rate is insufficient but the model has not been reverted")
                        self.nnet.save_checkpoint(
                            folder=self.args.checkpoint, filename=self.get_checkpoint_file(
                                i)
                        )
                else:
                    log.info("ACCEPTING NEW MODEL")
                    self.nnet.save_checkpoint(
                        folder=self.args.checkpoint, filename=self.get_checkpoint_file(
                            i)
                    )
                    self.nnet.save_checkpoint(
                        folder=self.args.checkpoint, filename="best.pth.tar"
                    )
            else:
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename=self.get_checkpoint_file(
                        i)
                )

    def make_train_example(self, sp: SelfPlayer) -> list[TrainExample]:
        while True:
            rnum = sp.prepare_next(0, False)

            if rnum == 0:
                continue
            elif rnum == 1:
                pis, win_rates = self.nnet.predict(
                    sp.get_boards_for_prediction(0, False))
                sp.receive_prediction(pis, win_rates, 0, False)
            elif rnum == 2:
                return sp.get_train_examples()

    def get_checkpoint_file(self, iteration: int) -> str:
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def get_symmetries(self, train_example: TrainExample) -> list[TrainExample]:
        board = train_example.canonical_board
        pi = train_example.pi
        assert (len(pi) == self.n**2+1)  # 1 for pass
        board = np.reshape(board, (self.n, self.n))
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l: list[TrainExample] = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l.append(TrainExample(newB, train_example.cur_player, np.array(
                    list(newPi.ravel()) + [pi[-1]]), train_example.v))
        return l

    # def save_train_examples(self, iteration: int):
    #     folder = self.args.checkpoint
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
    #     filename = os.path.join(
    #         folder, self.get_checkpoint_file(iteration) + ".examples"
    #     )
    #     with open(filename, "wb+") as f:
    #         Pickler(f).dump(self.train_examples_history)
    #     f.closed  # なにこれ

    # def load_train_examples(self):
    #     model_file = os.path.join(
    #         self.args.load_folder_file[0], self.args.load_folder_file[1]
    #     )
    #     examples_file = model_file + ".examples"
    #     if not os.path.isfile(examples_file):
    #         log.warning(
    #             f'File "{examples_file}" with train_examples not found!')
    #         r = input("Continue? [y|n]")
    #         if r != "y":
    #             sys.exit()
    #     else:
    #         log.info("File with train_examples found")
    #         with open(examples_file, "rb") as f:
    #             self.train_examples_history = Unpickler(f).load()
    #         log.info("Loading done!")

    #         self.skip_first_self_play = True
