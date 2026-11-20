"""Tests for Stargazing Tonight.

Everything runs through LocalRunner, which enforces manifest.yaml exactly as
the live runtime does, and `_fetch` is replaced so no test touches the network.
"""

from __future__ import annotations

import json
import urllib.error
from datetime import datetime, timedelta, timezone

import pytest
from pippa_agentapp_sdk import (
    LocalRunner,
    OutputShapeViolation,
    ServiceUnavailableError,
)

from app import (
    SITES,
    App,
    NoPlaceGiven,
    OutsideCoverage,
    PlaceNotFound,
    dark_window,
    moon_phase,
    sun_altitude,
)

# 16:00 UTC on 20 November 2026: daylight in Sheffield, dark within the hour.
NOW = datetime(2026, 11, 20, 16, 0, tzinfo=timezone.utc)

SHEFFIELD = {"lat": 53.383, "lon": -1.466}

GEOCODE_SHEFFIELD = json.dumps({"results": [{
    "name": "Sheffield", "admin1": "England", "country_code": "GB",
    "latitude": SHEFFIELD["lat"], "longitude": SHEFFIELD["lon"],
    "timezone": "Europe/London",
}]}).encode()

RESERVES = json.dumps([
    {"name": "Sunnybank Local Nature Reserve", "lat": "53.3733", "lon": "-1.4842"},
    {"name": "SRWT Carr House Meadows", "lat": "53.4400", "lon": "-1.5300"},
]).encode()


def hourly(cloud_for_hour) -> dict:
    """One location's forecast: two days of hours from midnight UTC."""
    start = datetime(2026, 11, 20, tzinfo=timezone.utc)
    times = [start + timedelta(hours=h) for h in range(48)]
    return {"hourly": {
        "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
        "cloud_cover": [cloud_for_hour(t) for t in times],
    }}


def forecast_for(url: str, cloud_by_index) -> bytes:
    """A multi-location answer with one series per requested latitude."""
    from urllib.parse import parse_qs, urlparse

    count = len(parse_qs(urlparse(url).query)["latitude"][0].split(","))
    return json.dumps([hourly(lambda t, i=i: cloud_by_index(i, t))
                       for i in range(count)]).encode()


def fake_fetch(*, geocode=GEOCODE_SHEFFIELD, reserves=RESERVES,
               cloud=lambda i, t: 10, fail_on=None):
    def fetch(url: str, timeout: float = 6.0) -> bytes:
        if fail_on and fail_on in url:
            raise TimeoutError("timed out")
        if "geocoding-api.open-meteo.com" in url:
            return geocode
        if "nominatim.openstreetmap.org" in url:
            if isinstance(reserves, Exception):
                raise reserves
            return reserves
        if "api.open-meteo.com/v1/forecast" in url:
            return forecast_for(url, cloud)
        if "api.postcodes.io" in url:
            return json.dumps({"status": 200, "result": {
                "postcode": "S1 2HE", "latitude": 53.381, "longitude": -1.470,
                "admin_district": "Sheffield"}}).encode()
        raise AssertionError(f"unexpected url {url}")

    return fetch


def run(inputs: dict, fetch=None, *, explicit: bool = False, now=NOW):
    instance = App()
    instance._fetch = fetch or fake_fetch()
    return LocalRunner(instance).call(
        "stars.tonight", inputs, granted_secrets={}, now=now,
        request_id="test-1", explicit=explicit,
    ).result


def test_the_manifest_is_valid():
    instance = App()
    assert instance.manifest.app_id == "hack.app_bc12"
    assert [o.id for o in instance.manifest.outcomes] == ["stars.tonight"]


def test_a_clear_night_recommends_the_nearest_dark_spot():
    result = run({"place": "Sheffield"})
    data = result.data
    assert data["place_used"] == "Sheffield, England"
    assert data["timezone_used"] == "Europe/London"
    assert data["good_night"] is True
    assert data["best_spot"] == "Surprise View, Peak District"
    assert 5 < data["best_spot_distance_km"] < 20
    assert data["best_spot_cloud_percent"] == 10
    assert data["cloud_threshold_percent"] == 30
    assert len(data["other_spots"]) == 5
    assert all("km" in s and "% cloud" in s for s in data["other_spots"])
    assert data["nearby_park"] == "SRWT Carr House Meadows"
    assert data["nearby_park_cloud_percent"] == 10
    assert "Surprise View" in result.narrative


def test_a_cloudy_nearest_spot_loses_to_a_clear_one_further_away():
    # Index 0 is the place itself, 1 is the nearest dark spot.
    def cloud(i, t):
        return 95 if i in (0, 1) else 5

    data = run({"place": "Sheffield"}, fake_fetch(cloud=cloud)).data
    assert data["best_spot"] != "Surprise View, Peak District"
    assert data["best_spot_cloud_percent"] == 5
    assert data["local_cloud_percent"] == 95
    assert data["good_night"] is True


def test_cloud_everywhere_is_not_a_good_night():
    data = run({"place": "Sheffield"}, fake_fetch(cloud=lambda i, t: 80)).data
    assert data["good_night"] is False
    assert data["best_spot_cloud_percent"] == 80


def test_on_a_cloudy_night_a_nearby_spot_beats_a_slightly_clearer_far_one():
    def cloud(i, t):
        return 60 if i == 1 else (50 if i == 6 else 90)

    data = run({"place": "Sheffield"}, fake_fetch(cloud=cloud)).data
    assert data["good_night"] is False
    assert data["best_spot"] == "Surprise View, Peak District"


def test_dublin_is_sent_to_an_irish_spot_not_across_the_sea():
    dublin = json.dumps({"results": [{
        "name": "Dublin", "admin1": "Leinster", "country_code": "IE",
        "latitude": 53.35, "longitude": -6.26, "timezone": "Europe/Dublin"}]}).encode()
    data = run({"place": "Dublin"}, fake_fetch(geocode=dublin)).data
    assert data["best_spot"] == "Sally Gap, Wicklow Mountains"
    assert data["timezone_used"] == "Europe/Dublin"
    assert not any("Anglesey" in s for s in data["other_spots"])


def test_the_clearest_hour_is_reported():
    def cloud(i, t):
        return 0 if t.hour == 22 else 20

    data = run({"place": "Sheffield"}, fake_fetch(cloud=cloud)).data
    assert data["best_time"] == "22:00"   # GMT == UK time in November


def test_the_backup_reserve_is_optional():
    data = run({"place": "Sheffield"},
               fake_fetch(reserves=TimeoutError("slow"))).data
    assert "nearby_park" not in data
    assert "nearby_park_distance_km" not in data
    assert data["best_spot"]


def test_a_backup_further_than_the_dark_spot_is_left_out():
    far = json.dumps([{"name": "Distant Reserve", "lat": "53.49", "lon": "-1.33"}]).encode()
    data = run({"place": "Sheffield"}, fake_fetch(reserves=far)).data
    assert "nearby_park" not in data


def test_no_reserves_found_leaves_the_backup_out():
    data = run({"place": "Sheffield"}, fake_fetch(reserves=b"[]")).data
    assert "nearby_park" not in data


def test_a_postcode_is_looked_up_directly():
    data = run({"place": "s1 2he"}).data
    assert data["place_used"] == "S1 2HE, Sheffield"


def test_an_unknown_place_says_which_place():
    with pytest.raises(PlaceNotFound) as failure:
        run({"place": "Narnia"}, fake_fetch(geocode=b'{"generationtime_ms": 0.1}'))
    assert failure.value.user_message == "I could not find Narnia in the UK or Ireland."


def test_a_place_abroad_is_refused_politely():
    paris = json.dumps({"results": [{
        "name": "Paris", "country_code": "FR", "latitude": 48.85,
        "longitude": 2.35, "timezone": "Europe/Paris"}]}).encode()
    with pytest.raises(OutsideCoverage) as failure:
        run({"place": "Paris"}, fake_fetch(geocode=paris))
    assert "Paris" in failure.value.user_message


def test_no_place_asks_for_one_even_when_picked_by_name():
    for explicit in (False, True):
        with pytest.raises(NoPlaceGiven) as failure:
            run({"place": ""}, explicit=explicit)
        assert "town or postcode" in failure.value.user_message
    with pytest.raises(NoPlaceGiven):
        run({})


def test_picked_by_name_answers_the_same():
    assert run({"place": "Sheffield"}, explicit=True).data == run({"place": "Sheffield"}).data


def test_a_weather_timeout_is_a_retryable_service_error():
    with pytest.raises(ServiceUnavailableError):
        run({"place": "Sheffield"}, fake_fetch(fail_on="api.open-meteo.com/v1/forecast"))


def test_a_weather_rate_limit_is_retryable():
    def fetch(url, timeout=6.0):
        if "forecast" in url:
            raise urllib.error.HTTPError(url, 429, "Too Many", {}, None)
        return fake_fetch()(url, timeout)

    from pippa_agentapp_sdk import RateLimitedError

    with pytest.raises(RateLimitedError):
        run({"place": "Sheffield"}, fetch)


def test_an_empty_forecast_is_an_error_not_a_clear_sky():
    def fetch(url, timeout=6.0):
        if "forecast" in url:
            return b"[]"
        return fake_fetch()(url, timeout)

    with pytest.raises(ServiceUnavailableError):
        run({"place": "Sheffield"}, fetch)


def test_a_distance_limit_is_respected():
    data = run({"place": "Sheffield", "max_distance_km": 30}).data
    assert data["best_spot_distance_km"] <= 30


def test_an_impossible_distance_limit_falls_back_to_the_nearest():
    result = run({"place": "Sheffield", "max_distance_km": 1})
    assert result.data["best_spot"] == "Surprise View, Peak District"
    assert "within 1 km" in result.narrative


def test_darkness_in_a_sheffield_november():
    start, end, fully = dark_window(NOW, SHEFFIELD["lat"], SHEFFIELD["lon"])
    assert fully is True
    # Nautical dusk around 17:30 GMT, nautical dawn around 06:20 GMT.
    assert datetime(2026, 11, 20, 17, 0, tzinfo=timezone.utc) <= start <= datetime(2026, 11, 20, 17, 50, tzinfo=timezone.utc)
    assert datetime(2026, 11, 21, 6, 0, tzinfo=timezone.utc) <= end <= datetime(2026, 11, 21, 6, 50, tzinfo=timezone.utc)


def test_never_fully_dark_in_a_shetland_midsummer():
    june = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
    _, _, fully = dark_window(june, 60.15, -1.15)
    assert fully is False


def test_the_sun_is_high_at_noon_and_low_at_midnight():
    noon = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
    midnight = datetime(2026, 6, 21, 0, 0, tzinfo=timezone.utc)
    assert 55 < sun_altitude(noon, 51.48, 0.0) < 65      # Greenwich, about 62
    assert sun_altitude(midnight, 51.48, 0.0) < -10


def test_moon_phases_on_known_dates():
    # Full moon 24 November 2026; new moon 9 November 2026.
    phase, lit = moon_phase(datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc))
    assert phase == "full moon" and lit >= 95
    phase, lit = moon_phase(datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc))
    assert phase == "new moon" and lit <= 5


def test_every_site_is_in_the_uk_or_ireland():
    assert len(SITES) >= 40
    for site in SITES:
        assert 49.8 <= site["lat"] <= 60.9, site["name"]
        assert -10.7 <= site["lon"] <= 1.8, site["name"]
        assert site["name"] and site["type"] and site["region"]


def test_the_runner_enforces_the_manifest(monkeypatch):
    from pippa_agentapp_sdk import Result

    monkeypatch.setattr(App, "handle", lambda self, c: Result(data={"surprise": 1}))
    with pytest.raises(OutputShapeViolation):
        run({"place": "Sheffield"})
