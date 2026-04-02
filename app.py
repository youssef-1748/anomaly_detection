from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from anomaly_detector import *

app = FastAPI()

# autoriser frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None
stream = None
location = "Unknown"


@app.post("/start")
def start_match(data: dict):
    global engine, stream, location

    location = data.get("location", "Unknown")

    # reset moteur (important)
    engine = RealtimeScoringEngine()

    player_id = "P07_Mbappe"

    # IA intacte
    history = generate_history(player_id, n=30)
    engine.register_player(player_id, history)

    engine.set_context(MatchContext(
        temperature=34.0,
        humidite=60.0,
        vent=10.0,
        fatigue_cumulee=0.7,
        jours_depuis_dernier=2
    ))

    stream = simulate_gps_stream(
        player_id,
        duration_s=600,
        interval_s=5,
        inject_anomaly_at=300
    )

    return {"status": "started", "location": location}


@app.get("/next")
def get_next():
    global stream, engine

    if stream is None:
        return {"error": "match not started"}

    try:
        snap = next(stream)
    except StopIteration:
        return {"done": True}

    result = engine.ingest(snap)

    if result is None:
        return {"waiting": True}

    return {
        "player": result.player_id,
        "time": result.timestamp,
        "score": result.anomaly_score,
        "niveau": result.niveau,
        "explication": result.explication,
        "alerte": result.alerte,
        "location": location
    }