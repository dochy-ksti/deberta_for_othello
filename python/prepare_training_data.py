from collections import deque
from typing import List

from .mcts_args import MctsArgs

from .train_example import TrainExample
import random


def choose_training_data(que: deque[list[TrainExample]], args: MctsArgs) -> tuple[list[TrainExample], bool]:
    # args.epochsは、学習全体で一つのデータを何回学習するか、を意味している。
    # できるだけバラバラにして混ぜながら少しずつ学習するのが良いはずである。
    # 過学習を避けるため、3～5 epochs 程度にするのが良いのではないかと思っている。
    #
    # 1イテレーションごとにSelfPlayによって作られたデータが加えられる。
    # これを1データと数える。1データはlist[TrainExample]であり、dequeに追加されていく。
    # データは古くなるとpop_leftされるが、増えるペースのほうが減るペースより早く、
    # データ全体としては増えていく傾向になる。
    # 現在のところ2データ追加時に1データ削除される。
    #
    # 最新 n イテレーションのデータをhistory_coreとする。
    # history_coreにいる間は他よりも学習率が高く設定される
    # history_coreが何イテレーションあるかは history_core_len で設定される。
    #
    # history_core_rateは学習全体に占めるhistory_coreでの学習割合が示されている。
    # history_core_rateが0.5ならば、history_coreで学習した後、
    # 削除されるまでにhistory_coreと同等程度の回数の学習が行われる。
    # なのでhistory_coreは0.5、それ以外も0.5の割合ということになる。
    #
    # 5 epochs で、history_core_rateが0.5とすると、history_coreにいる間に2.5回の学習機会が訪れるように
    # 乱数を割り振らなければいけない。
    # これをhistory_core_lenで割ると、
    # 1イテレーションで1データに対してどの程度の割合の学習が行われるかを求められる
    unit_learning_rate = args.epochs * args.history_core_rate / args.history_core_len
    if 1 < unit_learning_rate:
        raise ValueError(
            "unit_learning_rate is over 1.0. It's too much to learn.")

    # history_coreを抜けると選択率は線形に落ちていき、最終的に0になる。
    # その場合の直角三角形の面積は n*unit_learning_rate / 2 になる
    # history_coreの面積は history_core_len * unit_learning_rateであり、
    # 三角形とその四角形の比率は history_core_rate になるので、
    # (n*unit_learning_rate/2)*history_core_rate ==
    #  (history_core_len * unit_learning_rate)*(1-history_core_rate)
    # となる。
    # n*unit_learning_rate == 2*(history_core_len *
    #   unit_learning_rate)*(1-history_core_rate) / history_core_rate
    # n == 2*history_core_len*(1/history_core_rate - 1)
    # n を noncore_len とすると
    noncore_len = int(2 * args.history_core_len *
                      (1/args.history_core_rate - 1))

    if noncore_len == 0:
        raise ValueError("noncore_len == 0. history_core_rate is too high.")

    # noncore_len に足りる長さのデータが有れば、初期三角形が完成する。
    # 三角形完成後までデータは消えないので、学習を進めていけば初期三角形はいずれ完成する。
    is_triangle = args.history_core_len + noncore_len < len(que)

    # 初期三角形よりデータが長い場合、長さに応じた学習比率の変更が必要になる。
    # 初期三角形よりどれだけデータが長いかを求める。
    triangle_rate = (len(que) - args.history_core_len) / noncore_len

    # noncore_lenは初期三角形のlenであるが、現在の実際のnoncore_lenはこちら
    actual_noncore_len = len(que) - args.history_core_len

    gathered: List[TrainExample] = []

    for index, list in enumerate(reversed(que)):
        if index < args.history_core_len:
            gathered.extend(extract_data(list, unit_learning_rate))
        else:
            noncore_index = index - args.history_core_len
            if not is_triangle:
                # 初期三角形完成までは、選択率が線形に落ちていって、
                # 最後まで行ったときに予定の学習が完了するように比率を割り振る
                rate = unit_learning_rate * (
                    (noncore_len - noncore_index) / noncore_len)
                gathered.extend(extract_data(list, rate))
            else:
                rate = unit_learning_rate * (
                    actual_noncore_len - noncore_index) / actual_noncore_len

                # 三角形が長く伸びている場合、それに応じて比率を下げる
                gathered.extend(extract_data(list, rate / triangle_rate))

    return (gathered, is_triangle)


def extract_data(list: list[TrainExample], unit_learning_rate: float) -> list[TrainExample]:
    # いちおう重複なしで取ってくる
    return random.sample(list, k=int(len(list) * unit_learning_rate))
