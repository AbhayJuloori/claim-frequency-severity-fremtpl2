import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(artifacts, monkeypatch):
    import api.main as main
    monkeypatch.setattr(main, "ART", artifacts)
    main.models.cache_clear(); main.artifact.cache_clear()
    return TestClient(main.app)


POLICY = {"Area": "C", "VehBrand": "B12", "VehGas": "Regular", "Region": "R11", "VehPower": 6,
          "VehAge": 3, "DrivAge": 40, "BonusMalus": 50, "Density": 1200}


def test_health_and_metrics(client):
    assert client.get("/health").json() == {"ok": True}
    assert "pure_premium" in client.get("/metrics").json()


def test_segments_and_unknown_dimension(client):
    rows = client.get("/segments/DrivAgeBand").json()
    assert rows and {"segment", "lgb_relativity"} <= set(rows[0])
    assert client.get("/segments/Nope").status_code == 404
    assert "Area" in client.get("/segments-all").json()


def test_notes_hide_unverified_prose(client):
    n = client.get("/notes").json()[0]
    assert n["verified"] is False and n["note"] is None and n["problems"] == ["9.9"]


def test_predict_monotone_in_bonus_malus(client):
    worse = {**POLICY, "BonusMalus": 120}
    a, b = client.post("/predict", json=[POLICY, worse]).json()
    assert a["glm"]["pure_premium"] > 0 and b["glm"]["frequency"] > a["glm"]["frequency"]


def test_predict_validates_input(client):
    assert client.post("/predict", json=[{**POLICY, "DrivAge": 12}]).status_code == 422
