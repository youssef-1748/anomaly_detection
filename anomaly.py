# anomaly_detector.py

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ===================== DATA STRUCTURES =====================

@dataclass
class MatchContext:
    temperature: float
    humidite: float
    vent: float
    fatigue_cumulee: float
    jours_depuis_dernier: int
    minutes_jouees: float = 0.0


@dataclass
class GPSSnapshot:
    player_id: str
    timestamp: float
    vitesse: float
    acceleration: float
    distance_cumul: float
    nb_sprints_cumul: int
    fc: float


@dataclass
class AnomalyResult:
    player_id: str
    timestamp: float
    anomaly_score: float
    niveau: str
    alerte: bool
    z_scores: Dict[str, float] = field(default_factory=dict)
    explication: str = ""


# ===================== BASELINE =====================

class PlayerBaseline:
    FEATURE_COLS = [
        "vitesse_max", "vitesse_moy", "acceleration_max",
        "nb_sprints_per90", "distance_per90", "fc_moy"
    ]

    def __init__(self, player_id: str):
        self.player_id = player_id
        self.scaler = StandardScaler()
        self.model = None
        self.global_mean = None
        self.global_std = None
        self.trained = False

    def fit(self, history: pd.DataFrame):
        df = history.copy()

        if "minutes_joues" not in df.columns:
            df["minutes_joues"] = 90

        df["nb_sprints_per90"] = df["nb_sprints"] / df["minutes_joues"] * 90
        df["distance_per90"] = df["distance_totale"] / df["minutes_joues"] * 90

        self.global_mean = df[self.FEATURE_COLS].mean()
        self.global_std = df[self.FEATURE_COLS].std().replace(0, 1e-6)

        X = (df[self.FEATURE_COLS] - self.global_mean) / self.global_std
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(n_estimators=200, contamination=0.05)
        self.model.fit(X_scaled)

        self.trained = True


# ===================== CONTEXT =====================

class ContextCorrector:
    def correct(self, features: pd.Series, context: MatchContext):
        f = features.copy()

        temp_factor = 1 + 0.008 * max(0, context.temperature - 18)
        fatigue_factor = 1 + 0.18 * context.fatigue_cumulee

        f["vitesse_max"] *= temp_factor
        f["vitesse_moy"] *= temp_factor
        f["acceleration_max"] *= fatigue_factor

        return f


# ===================== AGGREGATOR =====================

class GPSAggregator:
    def __init__(self):
        self.buffer: Dict[str, List[GPSSnapshot]] = {}

    def push(self, snap: GPSSnapshot):
        self.buffer.setdefault(snap.player_id, []).append(snap)

    def aggregate(self, player_id: str, minutes_jouees: float):
        buf = self.buffer.get(player_id, [])
        if len(buf) < 5:
            return None

        return pd.Series({
            "vitesse_max": max(s.vitesse for s in buf),
            "vitesse_moy": np.mean([s.vitesse for s in buf]),
            "acceleration_max": max(s.acceleration for s in buf),
            "nb_sprints_per90": buf[-1].nb_sprints_cumul / minutes_jouees * 90,
            "distance_per90": buf[-1].distance_cumul / minutes_jouees * 90,
            "fc_moy": np.mean([s.fc for s in buf])
        })


# ===================== ENGINE =====================

class RealtimeScoringEngine:
    SEUIL_SUSPECT = 0.65
    SEUIL_CRITIQUE = 0.82

    def __init__(self):
        self.baselines = {}
        self.corrector = ContextCorrector()
        self.aggregator = GPSAggregator()
        self.context = None

    def register_player(self, player_id, history):
        bl = PlayerBaseline(player_id)
        bl.fit(history)
        self.baselines[player_id] = bl

    def set_context(self, context):
        self.context = context

    def ingest(self, snap: GPSSnapshot):
        self.context.minutes_jouees = snap.timestamp / 60
        self.aggregator.push(snap)

        bl = self.baselines[snap.player_id]

        features = self.aggregator.aggregate(
            snap.player_id, self.context.minutes_jouees
        )

        if features is None:
            return None

        features = self.corrector.correct(features, self.context)

        X = (features - bl.global_mean) / bl.global_std
        X_scaled = bl.scaler.transform([X])

        score = bl.model.score_samples(X_scaled)[0]
        anomaly_score = 1 / (1 + np.exp(-score * 5))

        if anomaly_score > self.SEUIL_CRITIQUE:
            niveau = "critique"
        elif anomaly_score > self.SEUIL_SUSPECT:
            niveau = "suspect"
        else:
            niveau = "normal"

        return AnomalyResult(
            player_id=snap.player_id,
            timestamp=snap.timestamp,
            anomaly_score=round(anomaly_score, 3),
            niveau=niveau,
            alerte=anomaly_score > self.SEUIL_SUSPECT,
            explication=f"Score {anomaly_score:.2f}"
        )


# ===================== SIMULATION =====================

def generate_history(player_id, n=30):
    rng = np.random.default_rng(0)

    return pd.DataFrame({
        "vitesse_max": rng.normal(29, 1, n),
        "vitesse_moy": rng.normal(8, 0.5, n),
        "acceleration_max": rng.normal(4, 0.3, n),
        "nb_sprints": rng.integers(15, 30, n),
        "distance_totale": rng.normal(10, 1, n),
        "fc_moy": rng.normal(150, 10, n),
        "minutes_joues": rng.normal(80, 10, n)
    })


def simulate_gps_stream(player_id, duration_s=600, interval_s=5, inject_anomaly_at=300):
    rng = np.random.default_rng(1)

    t = 0
    dist = 0
    sprints = 0

    while t <= duration_s:
        if t >= inject_anomaly_at:
            vitesse = rng.normal(33, 1)
            accel = rng.normal(5, 0.2)
            fc = rng.normal(180, 5)
        else:
            vitesse = rng.normal(29, 1)
            accel = rng.normal(4, 0.2)
            fc = rng.normal(150, 5)

        dist += vitesse * interval_s / 3600
        if vitesse > 25:
            sprints += 1

        yield GPSSnapshot(
            player_id, t, vitesse, accel, dist, sprints, fc
        )

        t += interval_s