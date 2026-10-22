from typing import Literal

from pydantic import BaseModel, Field


class InputData(BaseModel):
    """Physical measurements of one abalone (lengths in mm / 200, weights in grams / 200, as in the dataset)."""

    sex: Literal["M", "F", "I"] = Field(..., description="M (male), F (female) or I (infant)")
    length: float = Field(..., gt=0, description="Longest shell measurement")
    diameter: float = Field(..., gt=0, description="Perpendicular to length")
    height: float = Field(..., gt=0, description="With meat in shell")
    whole_weight: float = Field(..., gt=0, description="Whole abalone")
    shucked_weight: float = Field(..., gt=0, description="Weight of meat")
    viscera_weight: float = Field(..., gt=0, description="Gut weight (after bleeding)")
    shell_weight: float = Field(..., gt=0, description="After being dried")

    class Config:
        """Example displayed in the API documentation."""

        schema_extra = {
            "example": {
                "sex": "M",
                "length": 0.455,
                "diameter": 0.365,
                "height": 0.095,
                "whole_weight": 0.514,
                "shucked_weight": 0.2245,
                "viscera_weight": 0.101,
                "shell_weight": 0.15,
            }
        }


class PredictionOut(BaseModel):
    """Predicted number of rings (the age in years is rings + 1.5)."""

    rings_prediction: float
