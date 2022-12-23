import torch

from nucleotides.model.lightning_model import FunctionalModel


class PinkPanther(FunctionalModel):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv1d(4, 480, kernel_size=conv_kernel_size),
            torch.nn.ELU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            torch.nn.Dropout(p=0.2),
            torch.nn.Conv1d(480, 480, kernel_size=conv_kernel_size),
            torch.nn.ELU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            torch.nn.Dropout(p=0.2),
            torch.nn.Conv1d(480, 480, kernel_size=conv_kernel_size),
            torch.nn.ELU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
            torch.nn.Dropout(p=0.2),
            torch.nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(p=0.5),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.LazyLinear(512),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(512, 512),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(512, 512),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(512, self.hparams.n_targets),
        )

    def forward(self, x):
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), out.size(1) * out.size(2))
        predict = self.classifier(reshape_out)
        return predict
