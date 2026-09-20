from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

ROOT = Path(__file__).resolve().parent
PREDICTION_FILE = os.getenv("PREDICTION_FILE", "prediction_30_buildings.parquet")
PREDICTION_PATH = ROOT / "data" / PREDICTION_FILE
STREAM_DELAY_SECONDS = 30.0

app = FastAPI(title="Fault Detection API", version="1.0.0")
_prediction_df: pd.DataFrame | None = None
_timestamps: list[pd.Timestamp] | None = None
_timestamp_set: set[pd.Timestamp] | None = None
_timestamp_by_number: dict[int, pd.Timestamp] | None = None
_default_timestamp: pd.Timestamp | None = None


def _json_default(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _clean_record(record: dict[str, Any], mode: str) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in record.items():
        if pd.isna(value) or value is False:
            continue
        if hasattr(value, "item"):
            value = value.item()
        cleaned[key] = value

    if mode == "predictions":
        status = cleaned.get("status")
        if str(cleaned.get("predicted_fault_type", "")).lower() in {"", "none", "nan"}:
            cleaned.pop("predicted_fault_type", None)
        if status != "warning":
            cleaned.pop("hours_to_fault_prediction", None)
    return cleaned


def _load_predictions() -> pd.DataFrame:
    global _prediction_df, _timestamps, _timestamp_set, _timestamp_by_number, _default_timestamp
    if _prediction_df is None:
        if not PREDICTION_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "prediction_30_buildings.parquet is missing. Run: "
                    "python code/build_prediction_parquet.py"
                ),
            )
        frame = pd.read_parquet(PREDICTION_PATH)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.sort_values(["timestamp", "building_id"]).reset_index(drop=True)
        _prediction_df = frame
        _timestamps = sorted(frame["timestamp"].dropna().unique().tolist())
        _timestamp_set = set(_timestamps)
        _timestamp_by_number = {index: value for index, value in enumerate(_timestamps, start=1)}
        counts = frame.groupby("timestamp")["building_id"].nunique()
        max_count = counts.max()
        _default_timestamp = counts.loc[counts == max_count].index.max()
    return _prediction_df


def _all_timestamps() -> list[pd.Timestamp]:
    _load_predictions()
    return _timestamps or []


def _normalize_timestamp(timestamp: str | None) -> pd.Timestamp:
    timestamps = _all_timestamps()
    if not timestamps:
        raise HTTPException(status_code=404, detail="No timestamps found in prediction_30_buildings.parquet")
    if timestamp is None:
        return _default_timestamp or timestamps[-1]
    if str(timestamp).strip().isdigit():
        timestamp_number = int(str(timestamp).strip())
        if timestamp_number == 0:
            timestamp_number = 1
        value = (_timestamp_by_number or {}).get(timestamp_number)
        if value is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "Timestamp number not found",
                    "requested": timestamp_number,
                    "first_available_number": 1,
                    "last_available_number": len(timestamps),
                },
            )
        return value
    try:
        value = pd.Timestamp(timestamp)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {timestamp}") from exc
    if value not in (_timestamp_set or set(timestamps)):
        nearest = min(timestamps, key=lambda item: abs(item - value))
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Timestamp not found",
                "requested": value.isoformat(),
                "nearest_available": nearest.isoformat(),
            },
        )
    return value


def _records_for_timestamp(timestamp: pd.Timestamp, mode: str) -> dict[str, Any]:
    frame = _load_predictions()
    rows = frame.loc[frame["timestamp"] == timestamp].copy()
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No rows found for {timestamp.isoformat()}")

    reading_columns = [
        "building_id",
        "latitude",
        "longitude",
        "local_x_m",
        "local_y_m",
        "real_lat",
        "real_lng",
        "primaryspaceusage",
        "sub_primaryspaceusage",
        "sqm",
        "sqft",
        "yearbuilt",
        "numberoffloors",
        "meter_reading",
        "airTemperature",
    ]
    prediction_columns = reading_columns + [
        "status",
        "predicted_fault_type",
        "hours_to_fault_prediction",
    ]
    columns = reading_columns if mode == "readings" else prediction_columns
    payload_rows = [
        _clean_record(record, mode)
        for record in rows[columns].where(pd.notna(rows[columns]), None).to_dict(orient="records")
    ]
    timestamp_number = timestamps.index(timestamp) + 1 if timestamp in (timestamps := _all_timestamps()) else None
    return {
        "timestamp_number": timestamp_number,
        "timestamp": timestamp.isoformat(),
        "count": len(payload_rows),
        "items": payload_rows,
    }


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, default=_json_default, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


async def _stream(mode: str, delay_seconds: float, start_timestamp: str | None):
    timestamps = _all_timestamps()
    start = _normalize_timestamp(start_timestamp) if start_timestamp else timestamps[0]
    start_index = timestamps.index(start)
    for timestamp in timestamps[start_index:]:
        yield _sse_event(mode, _records_for_timestamp(timestamp, mode))
        await asyncio.sleep(delay_seconds)


@app.get("/health")
def health() -> dict[str, Any]:
    frame = _load_predictions()
    timestamps = _all_timestamps()
    return {
        "status": "ok",
        "prediction_path": str(PREDICTION_PATH),
        "rows": len(frame),
        "buildings": int(frame["building_id"].nunique()),
        "timestamps": len(timestamps),
        "first_timestamp_number": 1 if timestamps else None,
        "last_timestamp_number": len(timestamps) if timestamps else None,
        "first_timestamp": timestamps[0].isoformat() if timestamps else None,
        "last_timestamp": timestamps[-1].isoformat() if timestamps else None,
        "default_timestamp": (_default_timestamp.isoformat() if _default_timestamp is not None else None),
    }


@app.get("/readings")
def readings(timestamp: str | None = Query(default=None)) -> dict[str, Any]:
    return _records_for_timestamp(_normalize_timestamp(timestamp), "readings")


@app.get("/predictions")
def predictions(timestamp: str | None = Query(default=None)) -> dict[str, Any]:
    return _records_for_timestamp(_normalize_timestamp(timestamp), "predictions")


@app.get("/readings/stream")
def readings_stream(
    start_timestamp: str | None = Query(default=None),
    delay_seconds: float = Query(default=STREAM_DELAY_SECONDS, ge=0.0, le=60.0),
) -> StreamingResponse:
    return StreamingResponse(
        _stream("readings", delay_seconds, start_timestamp),
        media_type="text/event-stream",
    )


@app.get("/predictions/stream")
def predictions_stream(
    start_timestamp: str | None = Query(default=None),
    delay_seconds: float = Query(default=STREAM_DELAY_SECONDS, ge=0.0, le=60.0),
) -> StreamingResponse:
    return StreamingResponse(
        _stream("predictions", delay_seconds, start_timestamp),
        media_type="text/event-stream",
    )
