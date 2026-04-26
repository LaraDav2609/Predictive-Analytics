from datetime import datetime
from pydantic import BaseModel


class Driver(BaseModel):
    id: str
    number: int | None = None
    code: str
    first_name: str
    last_name: str
    nationality: str
    team: str
    profile_url: str | None = None
    photo_url: str | None = None
    points: float = 0.0
    wins: int = 0
    podiums: int = 0
    position: int | None = None  # championship position


class Constructor(BaseModel):
    id: str
    name: str
    nationality: str
    points: float = 0.0
    wins: int = 0
    position: int | None = None


class RacePrediction(BaseModel):
    driver_predictions: dict[str, "DriverRacePrediction"] = {}
    model_version: str = "f1-live-historical-sentiment-v2"
    generated_at: datetime | None = None
    data_sources: list[str] = []
    confidence: float | None = None


class DriverRacePrediction(BaseModel):
    driver_id: str
    driver_name: str
    win_prob: float
    podium_prob: float
    top5_prob: float
    predicted_position: int | None = None
    form_score: float | None = None
    team_score: float | None = None
    sentiment_score: float | None = None
    reliability_score: float | None = None
    confidence: float | None = None
    recent_summary: str | None = None
    explanation: list[str] = []


class Race(BaseModel):
    round: int
    name: str
    circuit: str
    country: str
    date: datetime
    status: str = "SCHEDULED"  # SCHEDULED, COMPLETED
    results: list["RaceResult"] = []
    prediction: RacePrediction | None = None


class RaceResult(BaseModel):
    position: int
    driver_id: str
    driver_name: str
    team: str
    time: str | None = None
    points: float = 0.0
    status: str = "Finished"
