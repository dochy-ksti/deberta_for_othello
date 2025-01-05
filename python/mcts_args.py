from dataclasses import dataclass


@dataclass
class MctsArgs:

    update_threshold: float = 0.55001
    do_arena: bool = False
    # Revert the model if rejected
    enable_rejecting: bool = False
    # if False, compare with a basically random player
    compare_with_self: bool = True

    use_deberta: bool = True
    checkpoint: str = "./temp/"
    load_model: bool = False
    load_folder_file: tuple[str, str] = (
        "/dev/models/8x100x50", "best.pth.tar")

    is_release: bool = True

    lr: float = 1e-4

    dropout: float = 0.1

    epochs: int = 5
    #num_iters_for_train_examples_history: int = 100

    # 直近 N イテレーション で生成されたデータを重点的に学習する
    history_core_len: int = 10

    # core部分の学習にどのくらいの割合を費やすか
    history_core_rate: float = 0.5

    # 毎回optimizerを作成する。optimizerの保存が不要になる。
    # 学習にデメリットがあるかもしれないが確認できていない。
    recreate_optimizer: bool = True

    cuda: bool = True
    num_channels: int = 512
