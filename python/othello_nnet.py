import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mcts_args import MctsArgs

from .intf_py_communicator import PyCommunicator


import logging


log = logging.getLogger(__name__)


class OthelloNNet(nn.Module):
    def __init__(self, pc: PyCommunicator, args: MctsArgs):
        self.board_x = pc.size_x()
        self.board_y = pc.size_y()
        self.action_size = pc.move_len()
        self.args = args
        log.info(args)

        super(OthelloNNet, self).__init__()  # type: ignore
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(
            args.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024
        )
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s: Tensor):
        s = s.view(-1, 1, self.board_x, self.board_y).contiguous().cuda()
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.args.num_channels *
                   (self.board_x - 4) * (self.board_y - 4))

        s = F.dropout(
            F.relu(self.fc_bn1(self.fc1(s))),
            p=self.args.dropout,
            training=self.training,
        )
        s = F.dropout(
            F.relu(self.fc_bn2(self.fc2(s))),
            p=self.args.dropout,
            training=self.training,
        )

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
