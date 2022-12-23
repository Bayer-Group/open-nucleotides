import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from nucleotides.model.sampler import NucleotidesDataset
from nucleotides.predict.inference_model import get_model
from nucleotides.predict.predict import Predictor
from nucleotides.predict.accumulator import ResultAccumulator


def run_evalutaion(model, n_mc_dropout=20):
    predictor = Predictor(model=model, n_mc_dropout=n_mc_dropout)
    dataloader = DataLoader(
        NucleotidesDataset(
            mode="validate",
            sequence_length=model.hparams.sequence_length,
            batch_size=32,
        ),
        batch_size=None,
        num_workers=8,
    )
    accumulator = ResultAccumulator()

    for features, target in dataloader:
        prediction, std = predictor._predict(features.transpose(1, 2).float())
        loss = (prediction.cpu() - target.float()).abs()
        accumulator.add_batch(
            {"loss": loss, "std": std.cpu(), "prediction": prediction}
        )

    aggregated = accumulator.aggregate()
    loss = aggregated["loss"].astype("float")
    std = aggregated["std"].astype("float")
    prediction = aggregated["prediction"].astype("float")

    correlation = feature_correlation2(loss, std)
    print(correlation)

    plot = sns.scatterplot(x=(np.abs(prediction[:, 0] - 0.5) - 0.5) * -1, y=std[:, 0])
    fig = plot.get_figure()
    fig.savefig("mc_dropout.png")


def feature_correlation2(matrix_a, matrix_b):
    correlations = []

    for column in range(matrix_a.shape[1]):
        column_a = matrix_a[:, column]
        column_b = matrix_b[:, column]
        correlations.append(pearsonr(column_a, column_b)[0])
    return correlations


if __name__ == "__main__":
    pretrained_model = get_model()
    pretrained_model.cuda(device=6)
    run_evalutaion(model=pretrained_model)
