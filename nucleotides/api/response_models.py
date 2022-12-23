from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class PredictedFeatureInfo(BaseModel):
    name: str
    experiment_target: str
    biosample_type: str
    biosample_term_name: str
    umap1: Union[float, str]
    umap2: Union[float, str]
    mean_prediction: Union[float, str]
    std_prediction: Union[float, str]

    class Config:
        schema_extra = {
            "example": {
                "name": "fibroblast_of_breast_h3k36me3",
                "experiment_target": "H3K36me3-human",
                "biosample_type": "primary cell",
                "biosample_term_name": "fibroblast of breast",
                "umap1": 21.223915,
                "umap2": 9.310385,
                "mean_prediction": 0.37686342,
                "std_prediction": 0.24983314,
            }
        }


class Prediction(BaseModel):
    prediction: float
    std: float


class VariantEffect(BaseModel):
    ref_predicition: float
    ref_std: float
    alt_predicition: float
    alt_std: float
    pvalue: float
    diff: float


class VariantEffects(BaseModel):
    predictions: Dict[str, VariantEffect]
    matches_reference_genome: bool
    contains_unknown_bases: bool


class OptimizationResult(BaseModel):
    sequences: List[str]
    optimization_curves: Optional[Dict[str, List[float]]]
