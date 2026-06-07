"""Driver and constructor metadata normalization."""

from __future__ import annotations

from sports.f1.models.f1 import Constructor, Driver


class MetadataFeatureProvider:
    def __init__(self, drivers: list[Driver], constructors: list[Constructor]):
        self._drivers = drivers
        self._constructors = constructors

    def get_features(self) -> dict:
        return {
            "drivers": {
                driver.id: {
                    "driver_id": driver.id,
                    "driver_name": f"{driver.first_name} {driver.last_name}".strip(),
                    "code": driver.code,
                    "team": driver.team,
                    "number": driver.number,
                    "nationality": driver.nationality,
                    "photo_url": driver.photo_url,
                    "profile_url": driver.profile_url,
                    "source": "f1_client",
                    "missing_data": False,
                }
                for driver in self._drivers
            },
            "constructors": {
                constructor.name.lower(): {
                    "constructor_id": constructor.id,
                    "team": constructor.name,
                    "nationality": constructor.nationality,
                    "points": constructor.points,
                    "position": constructor.position,
                    "source": "f1_client",
                    "missing_data": False,
                }
                for constructor in self._constructors
            },
        }
