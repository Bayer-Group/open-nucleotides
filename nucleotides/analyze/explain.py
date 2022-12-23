import altair as alt
import pandas as pd
import seaborn as sns
import torch
from captum.attr import IntegratedGradients
from umap import UMAP

from nucleotides.predict.inference_model import get_model
from nucleotides.util.util import onehot_sequence


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, index, flip_prediction=False):
        super().__init__()
        self.model = model
        self.index = index
        self.flip_prediction = flip_prediction

    def forward(self, X):
        X = self.model(X)
        X = torch.sigmoid(X[:, self.index])
        if self.flip_prediction:
            X = 1 - X
        return X


def explain(model, sequence, feature):
    input = onehot_sequence(sequence)
    input = input.unsqueeze(0)
    input.requires_grad_()
    input = input.to(device=model.device)
    index = model.hparams.target_names.index(feature)
    model = ModelWrapper(model, index)
    ig = IntegratedGradients(model)
    attr, delta = ig.attribute(
        input,
        target=None,
        return_convergence_delta=True,
        baselines=0.25,
    )
    attr = attr.detach().cpu().squeeze()
    summed = attr.abs().sum(dim=0)
    print(summed.shape)
    return summed.numpy()


def explain_multiple(
        model,
        sequence,
        endpoints=None,
        negative_endpoints=None,
        organize=True,
):
    if not endpoints and not negative_endpoints:
        raise ValueError(
            "At least one of endpoints or negative_endpoints should be provided"
        )
    features = model.hparams.target_names

    if endpoints == "all":
        endpoints = features
        negative_endpoints = []
    else:
        endpoints = endpoints or []
        negative_endpoints = negative_endpoints or []
    flips = [False] * len(endpoints) + [True] * len(negative_endpoints)
    all_endpoints = endpoints + negative_endpoints

    explanations = []
    for endpoint, flip in zip(all_endpoints, flips):
        input = onehot_sequence(sequence)
        input = input.unsqueeze(0)
        input.requires_grad_()
        input = input.to(device=model.device)

        index = features.index(endpoint)
        wrapped_model = ModelWrapper(model, index, flip_prediction=flip)
        ig = IntegratedGradients(wrapped_model)
        attr, delta = ig.attribute(
            input,
            target=None,
            return_convergence_delta=True,
            baselines=0.25,
        )
        attr = attr.detach().cpu().squeeze()
        summed = attr.abs().sum(dim=0)
        # summed = attr.sum(dim=0)
        explanations.append(summed)
    explanations = torch.stack(explanations)
    explantions = pd.DataFrame(explanations.numpy(), index=endpoints)
    if organize:
        explantions = organize_explanation_matrix(explantions)
    return explantions


def organize_explanation_matrix(df):
    reducer = UMAP(n_components=1)
    nice_order = list(
        pd.DataFrame(reducer.fit_transform(df), index=df.index).sort_values(by=0).index,
    )
    reordered = df.reindex(nice_order)
    return reordered


def create_heatmap_altair(df):
    df = df.copy()
    df["endpoint_index"] = list(range(len(df)))
    long_table = df.melt(id_vars=["endpoint_index"])
    long_table.rename(columns={"variable": "sequence position"}, inplace=True)
    chart = (
        alt.Chart(long_table)
        .mark_rect()
        .encode(x="sequence position", y="endpoint_index", color="value")
    )
    chart.save("heatmap.html", embed_options={"renderer": "svg"})


def create_heatmap_seaborn(df):
    ax = sns.heatmap(df)
    fig = ax.get_figure()
    fig.savefig("heatmap2.png")


def create_heatmap_plotly(df, organize=True):
    if organize:
        df = organize_explanation_matrix(df)
    import plotly.express as px

    fig = px.imshow(
        df,
        color_continuous_scale=px.colors.diverging.BrBG,
        color_continuous_midpoint=0,
    )
    fig.update_layout(xaxis={"tickformat": "d"})
    return fig


if __name__ == "__main__":
    DEVICE = torch.device("cuda", 13)
    dummy_sequence = "ATTTTTCCAT" * 100
    model = get_model().eval().to(DEVICE)
    print("get attributions")
    explanations = explain_multiple(model, dummy_sequence, endpoints="all")
    explanations = organize_explanation_matrix(explanations)
    print(explanations)
    create_heatmap_plotly(explanations)
