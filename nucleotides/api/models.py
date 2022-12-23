from pydantic import BaseModel, Field, constr
from pydantic.typing import List, Literal, Optional, Union

from nucleotides.optimize_sequence.positive_and_negative_endpoints import (
    get_negative_features,
    get_positive_features,
)
from nucleotides.util.util import get_target_names

distinct_features = get_target_names()


class Variant(BaseModel):
    id: str
    chrom: str
    pos: int
    ref: str
    alt: str
    strand: str = Field(default="+")
    assembly: str

    class Config:
        schema_extra = {
            "example": {
                "id": "rs555418780",
                "chrom": "chr2",
                "pos": 1012621,
                "ref": "A",
                "alt": "T",
                "strand": "+",
                "reference_genome": "GRCh37",
            }
        }


class Region(BaseModel):
    chrom: str
    start: int
    end: Optional[int] = Field(alias="stop")
    strand: str = Field(default="+")
    reference_genome: Literal["GRCh37", "GRCh38"] = Field(
        default="GRCh38",
        alias="assembly",
    )

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "chrom": "chr2",
                "start": 40000000,
                "end": 40001000,
                "strand": "+",
                "assembly": "GRCh38",
            }
        }


class Position(BaseModel):
    chrom: str
    start: int
    strand: str = Field(default="+")
    assembly: Literal["GRCh37", "GRCh38"] = Field(default="GRCh38")

    class Config:
        schema_extra = {
            "example": {
                "chrom": "chr2",
                "start": 40000000,
                "strand": "+",
                "assembly": "GRCh38",
            }
        }


class PredictedFeatureInRegion(BaseModel):
    region: Region
    predicted_feature: str

    class Config:
        schema_extra = {
            "example": {
                "region": {
                    "chrom": "chr2",
                    "start": 40000000,
                    "end": 40001000,
                    "strand": "+",
                    "assembly": "GRCh38",
                },
                "predicted_feature": "fibroblast_of_breast_h3k36me3",
            }
        }


example_sequence = "CAGAGGCCTG" * 100

Sequence = constr(min_length=1000, max_length=10000, regex="^[CAGTcagt]+$")


class OptimizationParameters(BaseModel):
    sequences: Union[Sequence, List[Sequence]]
    positive_features: Optional[List[str]]
    negative_features: Optional[List[str]]
    other_features: Literal["mask", "positive", "negative"] = Field(default="mask")
    n_generations: int = Field(default=2000, gt=0, le=10000)
    penalize_distance: bool = True
    population_size: int = Field(default=1024, gt=2, le=10000)
    mutation_rate: float = Field(default=0.005, gt=0, le=1)
    n_tournaments: int = Field(default=3, gt=0, le=20)
    auto_balance_loss: bool = True
    only_return_top_n: int = Field(default=1, ge=1)
    return_optimization_curves = False

    class Config:
        schema_extra = {
            "example": {
                "sequences": [example_sequence, example_sequence],
                "positive_features": get_positive_features(distinct_features),
                "negative_features": get_negative_features(distinct_features),
                "other_features": "mask",
                "n_generations": 2000,
                "penalize_distance": True,
                "population_size": 1024,
                "mutation_rate": 0.005,
                "n_tournaments": 3,
                "auto_balance_loss": True,
                "only_return_top_n": 1,
                "return_optimization_curves": False,
            }
        }
