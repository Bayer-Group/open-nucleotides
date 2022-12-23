from typing import Dict, List

import fsspec
import pandas as pd
import torch
from fastapi import Body, FastAPI

from nucleotides import __version__ as version
from nucleotides import config
from nucleotides.analyze.explanations_for_region import explain_sequence
from nucleotides.api.models import (
    OptimizationParameters,
    PredictedFeatureInRegion,
    Region,
    Sequence,
    Variant,
    example_sequence,
)
from nucleotides.api.response_models import (
    OptimizationResult,
    PredictedFeatureInfo,
    VariantEffects,
)
from nucleotides.optimize_sequence.optimize import optimize_interactive
from nucleotides.predict.dataloader import get_encoded_sequences
from nucleotides.predict.inference_model import get_model
from nucleotides.predict.lookup_prediction import lookup_effect_predictions_in_region
from nucleotides.predict.predict import Predictor
from nucleotides.util.select_device import NoFreeDevicevailableError, select_device
from nucleotides.util.sequence_from_genome import get_nucleotide_sequence
from nucleotides.util.util import get_endpoint_description, ints_to_sequence

reference_genomes = {"GRCh37": config.GRCh37, "GRCh38": config.GRCh38}

description = """
Predictions of Regulatory elements in the genome.

"""

app = FastAPI(
    title="Nucleotides",
    description=description,
    version=version,
    contact={
        "name": "Joren Retel",
        "email": "joren.retel@bayer.com",
    },
)

device = select_device()
if not device:
    raise NoFreeDevicevailableError(config.DEVICE)
print(f"Using device: {device}")

predictor_mc_dropout = Predictor(
    model=get_model(), device=device, dataparallel=False, n_mc_dropout=100
)
predictor = Predictor(
    model=get_model(), device=device, dataparallel=False, n_mc_dropout=None
)
target_names = predictor.model.hparams.target_names
target_description = get_endpoint_description().fillna("undefined")
target_description.columns = [
    column_name.lower().replace(" ", "_")
    for column_name in target_description.columns
]
target_description.index.name = "name"
target_description.reset_index(inplace=True)


@app.get("/")
def read_root():
    return {"Nucleotides API Version": version}


@app.get("/predicted_features", tags=["meta data"], response_model=List[str])
def predicted_features(include_aggregation_options: bool = False):
    if include_aggregation_options:
        return [
                   "aggregate: avg",
                   "aggregate: avg(abs)",
                   "aggregate: max ref",
                   "aggregate: max alt",
                   "aggregate: max difference (abs)",
                   "aggregate: max increase",
                   "aggregate: max decrease",
               ] + target_names
    else:
        return target_names


@app.get(
    "/predicted_features_description",
    response_model=List[PredictedFeatureInfo],
    tags=["meta data"],
)
def predicted_features_description():
    return target_description.to_dict(orient="records")


@app.post("/sequence")
def get_sequence(region: Region):
    return get_nucleotide_sequence(**region.dict())


@app.post(
    "/predictions_for_sequences",
    response_model=List[Dict[str, float]],
    tags=["predict presence of (epi-)genetic features in a sequence"],
)
def predictions_for_sequences(
        sequences: List[Sequence] = Body(example=[example_sequence]),
):
    return predictor.predict_from_sequences(sequences).to_dict(orient="records")


@app.post(
    "/predictions_for_sequence",
    response_model=Dict[str, float],
    tags=["predict presence of (epi-)genetic features in a sequence"],
)
def predictions_for_sequence(sequence: Sequence = Body(example=example_sequence)):
    return predictor.predict_from_sequence(sequence).to_dict()


@app.post(
    "/predictions_for_positions",
    response_model=List[Dict[str, float]],
    tags=["predict presence of (epi-)genetic features in a sequence"],
)
def predictions_for_positions(positions: List[Region]):
    positions = [position.dict() for position in positions]
    print(positions)
    return predictor.predict_from_positions(positions).to_dict(orient="records")


@app.post(
    "/predictions_for_position",
    response_model=Dict[str, float],
    tags=["predict presence of (epi-)genetic features in a sequence"],
)
def predictions_for_position(position: Region):
    return predictor.predict_from_position(**position.dict()).to_dict()


@app.post(
    "/variant_effect_prediction",
    response_model=VariantEffects,
    tags=[
        "predict effect of variants by comparing predictions for REF and ALT sequences"
    ],
)
async def variant_effect_prediction(variant: Variant):
    reference_sequence_path = reference_genomes[variant.assembly]
    ref, alt, contains_unknown, match = get_encoded_sequences(
        variant.chrom,
        variant.pos,
        variant.id,
        variant.ref,
        variant.alt,
        strand=variant.strand,
        sequence_length=1000,
        reference_sequence_path=fsspec.open(reference_sequence_path),
    )
    predictions = predictor_mc_dropout.predict_variant_effect(
        ref=torch.from_numpy(ref).unsqueeze(0), alt=torch.from_numpy(alt).unsqueeze(0)
    )
    # remove batch dimension
    predictions = {key: value[0] for key, value in predictions.items()}
    predictions = to_effect_dict(predictions)
    effect = {
        "predictions": predictions,
        "matches_reference_genome": match,
        "contains_unknown_bases": contains_unknown,
    }
    return effect


@app.post(
    "/variant_effect_prediction_in_region",
    tags=[
        "predict effect of variants by comparing predictions for REF and ALT sequences"
    ],
)
async def variant_effect_prediction_in_region(
        feature_in_region: PredictedFeatureInRegion,
):
    feature = feature_in_region.predicted_feature
    region = feature_in_region.region
    (
        variant_data,
        ref_predictions,
        alt_predictions,
    ) = lookup_effect_predictions_in_region(
        config.PREDICTIONS, chrom=region.chrom, start=region.start, end=region.end
    )
    if variant_data.empty:
        return []
    if feature == "aggregate: avg":
        difference = (alt_predictions - ref_predictions).mean(axis=1)
        ref_predictions = ref_predictions.mean(axis=1)
        alt_predictions = alt_predictions.mean(axis=1)
    elif feature == "aggregate: avg(abs)":
        difference = (alt_predictions - ref_predictions).abs().mean(axis=1)
        ref_predictions = ref_predictions.mean(axis=1)
        alt_predictions = alt_predictions.mean(axis=1)
    elif feature == "aggregate: max ref":
        ids = ref_predictions.idxmax(axis=1)
        ref_predictions = ref_predictions.lookup(ids.index, ids)
        alt_predictions = alt_predictions.lookup(ids.index, ids)
        difference = alt_predictions - ref_predictions
        feature = ids
    elif feature == "aggregate: max alt":
        ids = alt_predictions.idxmax(axis=1)
        ref_predictions = ref_predictions.lookup(ids.index, ids)
        alt_predictions = alt_predictions.lookup(ids.index, ids)
        difference = alt_predictions - ref_predictions
        feature = ids
    elif feature == "aggregate: max difference (abs)":
        ids = (alt_predictions - ref_predictions).abs().idxmax(axis=1)
        ref_predictions = ref_predictions.lookup(ids.index, ids)
        alt_predictions = alt_predictions.lookup(ids.index, ids)
        difference = alt_predictions - ref_predictions
        feature = ids
    elif feature == "aggregate: max increase":
        ids = (alt_predictions - ref_predictions).idxmax(axis=1)
        ref_predictions = ref_predictions.lookup(ids.index, ids)
        alt_predictions = alt_predictions.lookup(ids.index, ids)
        difference = alt_predictions - ref_predictions
        feature = ids
    elif feature == "aggregate: max decrease":
        ids = (alt_predictions - ref_predictions).idxmin(axis=1)
        ref_predictions = ref_predictions.lookup(ids.index, ids)
        alt_predictions = alt_predictions.lookup(ids.index, ids)
        difference = alt_predictions - ref_predictions
        feature = ids
    elif feature in target_names:
        ref_predictions = ref_predictions[feature]
        alt_predictions = alt_predictions[feature]
        difference = alt_predictions - ref_predictions
    else:
        return None
    variant_data["ref_prediction"] = ref_predictions
    variant_data["alt_prediction"] = alt_predictions
    variant_data["difference"] = difference
    variant_data["feature"] = feature

    return variant_data.to_dict(orient="records")


@app.post(
    "/explanations_for_sequence",
    response_model=Dict[str, List[float]],
    tags=["Explainability"],
)
def explanations_for_sequence(sequence: Sequence = Body(example=example_sequence)):
    explanation = explain_sequence(predictor=predictor, sequence=sequence).transpose()
    explanation = explanation.astype(float).round(3)
    explanation = explanation.to_dict(orient="list")
    return explanation


@app.post(
    "/optimize_sequences",
    tags=["Optimization"],
    response_model=OptimizationResult,
)
def optimize_sequences(optimization_params: OptimizationParameters):
    params = optimization_params
    optimized = optimize_interactive(
        model=predictor.model,
        starting_population=params.sequences,
        positive_features=params.positive_features,
        negative_features=params.negative_features,
        other_features=params.other_features,
        n_generations=params.n_generations,
        population_size=params.population_size,
        mutation_rate=params.mutation_rate,
        n_tournaments=params.n_tournaments,
        auto_balance_loss=params.auto_balance_loss,
        penalize_distance=params.penalize_distance,
    )

    population = optimized.population.cpu().numpy()[: params.only_return_top_n]
    sequences = [ints_to_sequence(encoded) for encoded in population]
    if params.return_optimization_curves:
        optimization_curves = pd.DataFrame(
            optimized.best_predictions,
            columns=target_names,
        )
        optimization_curves.index.name = "iteration"
        optimization_curves = optimization_curves.to_dict(orient="list")
    else:
        optimization_curves = None

    return {"sequences": sequences, "optimization_curves": optimization_curves}


def to_effect_dict(effect, feature_names=target_names):
    effect_dict = {}
    for i, feature in enumerate(feature_names):
        effect_dict[feature] = {
            metric_name: vector[i].item() for metric_name, vector in effect.items()
        }
    return effect_dict


if __name__ == "__main__":
    # launching API for debugging.
    import uvicorn

    uvicorn.run(
        "main:app",
        host="localhost",
        port=8893,
        log_level="info",
        workers=10,
        debug=True,
    )
