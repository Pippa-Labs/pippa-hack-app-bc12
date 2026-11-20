"""Starspy.

Given a town or postcode in the UK or Ireland, find where to go stargazing
tonight: the nearest known dark-sky place with clear skies, a nearby nature
reserve as a closer (but brighter) backup, when it is properly dark, and what
the moon is doing.

Where the facts come from:

* Darkness   - a curated list of dark-sky places shipped beside this file
               (dark_sky_sites.json). No free service measures light
               pollution, so the list is the honest way to say "dark".
* Location   - postcodes.io for UK postcodes, Open-Meteo geocoding for names.
* Backup     - Nominatim (OpenStreetMap) "nature reserve" search near the
               place. Optional: if it fails, the answer goes on without it.
* Cloud      - one Open-Meteo forecast request for every spot at once.
* Sun, moon  - worked out here with standard formulas; no network.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from zoneinfo import ZoneInfo

from pippa_agentapp_sdk import (
    AgentAppError,
    Call,
    Manifest,
    RateLimitedError,
    Result,
    ServiceUnavailableError,
    load_manifest_file,
    validate_manifest,
)

MANIFEST_PATH = Path(__file__).with_name("manifest.yaml")
SITES = json.loads(Path(__file__).with_name("dark_sky_sites.json").read_text("utf-8"))

#: Nominatim refuses requests without a descriptive User-Agent.
USER_AGENT = "pippa-agent-app-stargazing-tonight/0.1 (https://github.com/Pippa-Labs)"

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search?"
POSTCODE_URL = "https://api.postcodes.io/postcodes/"
OUTCODE_URL = "https://api.postcodes.io/outcodes/"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast?"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search?"

COVERED_COUNTRIES = {"GB": "Europe/London", "IE": "Europe/Dublin"}

#: A spot counts as "clear" when its average cloud over the dark hours is
#: below this. Returned with every answer so the rule is never hidden.
CLOUD_THRESHOLD = 30

#: How many of the nearest dark-sky places to check the forecast for.
CANDIDATES = 6

#: The sun this far below the horizon or more counts as "properly dark"
#: (nautical twilight over). -18 would be ideal, but in a Scottish summer the
#: sun never gets that low, and -12 is already dark enough for the Milky Way
#: away from towns.
DARK_SUN_ALTITUDE = -12.0

#: Backup spot: look for nature reserves within this box around the place.
PARK_BOX_DEG_LAT = 0.14   # about 15 km
PARK_BOX_DEG_LON = 0.23   # about 15 km at UK latitudes

FULL_POSTCODE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$")
OUTCODE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?$")

#: A known new moon, used as the start of the lunar cycle.
NEW_MOON_EPOCH = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
SYNODIC_MONTH = 29.530588853


# ── Errors a person may see ─────────────────────────────────────────────────


class NoPlaceGiven(AgentAppError):
    user_message = (
        "Tell me a town or postcode in the UK or Ireland, for example "
        "\"where can I go stargazing near York\"."
    )
    retryable = False


class PlaceNotFound(AgentAppError):
    user_message = "I could not find that place in the UK or Ireland."
    retryable = False


class OutsideCoverage(AgentAppError):
    user_message = "I only know stargazing spots in the UK and Ireland."
    retryable = False


# ── Sun and moon ────────────────────────────────────────────────────────────


def sun_altitude(moment: datetime, lat: float, lon: float) -> float:
    """Sun's altitude in degrees above the horizon (negative below).

    The standard low-precision solar position (good to about a hundredth of a
    degree around now), which is far more than "is it dark yet" needs.
    """
    d = (moment - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400
    g = math.radians((357.529 + 0.98560028 * d) % 360)
    q = (280.459 + 0.98564736 * d) % 360
    ecliptic_lon = math.radians(q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    tilt = math.radians(23.439 - 0.00000036 * d)
    right_ascension = math.atan2(math.cos(tilt) * math.sin(ecliptic_lon), math.cos(ecliptic_lon))
    declination = math.asin(math.sin(tilt) * math.sin(ecliptic_lon))
    sidereal_deg = ((18.697374558 + 24.06570982441908 * d) % 24) * 15
    hour_angle = math.radians(sidereal_deg + lon) - right_ascension
    phi = math.radians(lat)
    altitude = math.asin(
        math.sin(phi) * math.sin(declination)
        + math.cos(phi) * math.cos(declination) * math.cos(hour_angle)
    )
    return math.degrees(altitude)


def dark_window(now: datetime, lat: float, lon: float) -> tuple[datetime, datetime, bool]:
    """Tonight's stretch of darkness, as (start, end, fully_dark), in UTC.

    "Tonight" is the first run of dark sky that still has at least an hour to
    go, so asking at 5am gets the coming night rather than the last forty
    minutes of this one. When the sun never gets low enough (a Scottish
    midsummer), the darkest stretch below -6 degrees is used instead, and
    `fully_dark` is False.
    """
    step = timedelta(minutes=10)
    start_scan = now.astimezone(timezone.utc).replace(second=0, microsecond=0)
    times = [start_scan + step * i for i in range(int(36 * 60 / 10))]

    for threshold, fully in ((DARK_SUN_ALTITUDE, True), (-6.0, False)):
        run_start = None
        for moment in times:
            dark = sun_altitude(moment, lat, lon) <= threshold
            if dark and run_start is None:
                run_start = moment
            elif not dark and run_start is not None:
                if moment - max(run_start, now) >= timedelta(hours=1):
                    return run_start, moment, fully
                run_start = None
        if run_start is not None:
            return run_start, times[-1], fully

    # Only reachable inside the Arctic summer, which is not this app's map.
    return start_scan, start_scan + timedelta(hours=1), False


def moon_phase(moment: datetime) -> tuple[str, int]:
    """(phase name, percent of the disc lit) at a moment."""
    age = ((moment - NEW_MOON_EPOCH).total_seconds() / 86400) % SYNODIC_MONTH
    lit = round((1 - math.cos(2 * math.pi * age / SYNODIC_MONTH)) / 2 * 100)
    names = [
        (1.85, "new moon"),
        (5.54, "waxing crescent"),
        (9.23, "first quarter"),
        (12.92, "waxing gibbous"),
        (16.61, "full moon"),
        (20.30, "waning gibbous"),
        (23.99, "last quarter"),
        (27.68, "waning crescent"),
    ]
    for limit, name in names:
        if age < limit:
            return name, lit
    return "new moon", lit


def on_island_of_ireland(site: dict) -> bool:
    """Irish and Northern Irish sites are the ones in a "County ..." region."""
    return site["region"].startswith("County ")


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (haversine)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


# ── The app ─────────────────────────────────────────────────────────────────


class App:
    def __init__(self) -> None:
        self._manifest = validate_manifest(load_manifest_file(MANIFEST_PATH))

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    def _fetch(self, url: str, timeout: float = 6.0) -> bytes:
        """Every outbound request goes through here (tests replace it)."""
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def handle(self, call: Call) -> Result:
        if call.outcome_id != "stars.tonight":
            raise AgentAppError(f"this app cannot do {call.outcome_id!r}")

        place = (call.inputs.get("place") or "").strip()
        if not place:
            raise NoPlaceGiven("no place in the message")
        max_km = call.inputs.get("max_distance_km")

        now: datetime = call.context.now
        lat, lon, place_used, zone_name, irish = self._locate(place)
        zone = ZoneInfo(zone_name)

        # The nearest dark-sky places on the same island (a straight line
        # from Dublin to Anglesey crosses the sea), within the distance asked
        # for if any.
        same_island = [s for s in SITES if on_island_of_ireland(s) == irish]
        ranked = sorted(
            same_island, key=lambda s: distance_km(lat, lon, s["lat"], s["lon"])
        )
        widened = False
        if max_km:
            within = [s for s in ranked if distance_km(lat, lon, s["lat"], s["lon"]) <= max_km]
            if within:
                ranked = within
            else:
                widened = True
        candidates = ranked[:CANDIDATES]

        park = self._nearby_reserve(lat, lon)

        spots = [(lat, lon)] + [(s["lat"], s["lon"]) for s in candidates]
        if park:
            spots.append((park["lat"], park["lon"]))
        clouds = self._cloud_forecasts(spots)

        start, end, fully_dark = dark_window(now, lat, lon)
        middle = start + (end - start) / 2
        phase, lit = moon_phase(middle)

        def summarise(series: list[tuple[datetime, int]]) -> tuple[int, datetime]:
            during = [(t, c) for t, c in series if start - timedelta(minutes=30) <= t <= end]
            if not during:
                # Forecast hours did not land inside a short window: use the
                # hour nearest its start rather than inventing a figure.
                during = [min(series, key=lambda tc: abs(tc[0] - start))]
            average = round(sum(c for _, c in during) / len(during))
            clearest = min(during, key=lambda tc: tc[1])[0]
            return average, clearest

        local_cloud, _ = summarise(clouds[0])
        scored = []
        for site, series in zip(candidates, clouds[1:1 + len(candidates)]):
            average, clearest = summarise(series)
            scored.append({
                "site": site,
                "km": round(distance_km(lat, lon, site["lat"], site["lon"]), 1),
                "cloud": average,
                "clearest": clearest,
            })

        # The nearest clear spot wins. If none is clear, the nearest one that
        # is nearly as clear as the clearest (within 10 points): a long drive
        # for 52% cloud instead of 58% is not worth recommending.
        clear = [s for s in scored if s["cloud"] < CLOUD_THRESHOLD]
        if clear:
            best = clear[0]
        else:
            lowest = min(s["cloud"] for s in scored)
            best = next(s for s in scored if s["cloud"] <= lowest + 10)
        good_night = best["cloud"] < CLOUD_THRESHOLD

        def local(moment: datetime) -> str:
            return moment.astimezone(zone).strftime("%H:%M")

        data = {
            "place_used": place_used,
            "timezone_used": zone_name,
            "dark_from": local(max(start, now)),
            "dark_until": local(end),
            "fully_dark": fully_dark,
            "moon_phase": phase,
            "moon_illumination_percent": lit,
            "local_cloud_percent": local_cloud,
            "good_night": good_night,
            "cloud_threshold_percent": CLOUD_THRESHOLD,
            "best_spot": best["site"]["name"],
            "best_spot_type": best["site"]["type"],
            "best_spot_distance_km": best["km"],
            "best_spot_cloud_percent": best["cloud"],
            "best_time": local(max(best["clearest"], start, now)),
            "other_spots": [
                f"{s['site']['name']} ({s['site']['type']}), {s['km']:g} km, "
                f"{s['cloud']}% cloud"
                for s in scored if s is not best
            ],
        }
        if park and distance_km(lat, lon, park["lat"], park["lon"]) < best["km"]:
            park_cloud, _ = summarise(clouds[-1])
            data["nearby_park"] = park["name"]
            data["nearby_park_distance_km"] = round(
                distance_km(lat, lon, park["lat"], park["lon"]), 1)
            data["nearby_park_cloud_percent"] = park_cloud

        return Result(data=data, narrative=self._sentence(data, widened, max_km))

    # ── Steps ───────────────────────────────────────────────────────────────

    def _locate(self, place: str) -> tuple[float, float, str, str, bool]:
        """(lat, lon, name used, IANA zone, on the island of Ireland)."""
        cleaned = " ".join(place.split())
        upper = cleaned.upper()

        if FULL_POSTCODE.match(upper) or OUTCODE.match(upper):
            is_full = bool(FULL_POSTCODE.match(upper))
            base = POSTCODE_URL if is_full else OUTCODE_URL
            raw = self._get(base + quote(upper.replace(" ", ""), safe=""), 4.0,
                            not_found=place)
            result = (json.loads(raw) or {}).get("result")
            if not result:
                self._not_found(place)
            district = result.get("admin_district")
            if isinstance(district, list):
                district = district[0] if district else None
            label = result.get("postcode") or result.get("outcode") or upper
            name = f"{label}, {district}" if district else label
            irish = upper.startswith("BT")   # Northern Ireland postcodes
            return float(result["latitude"]), float(result["longitude"]), name, "Europe/London", irish

        query = urlencode({"name": cleaned, "count": 10, "language": "en", "format": "json"})
        raw = self._get(GEOCODE_URL + query, 4.0, not_found=place)
        results = (json.loads(raw) or {}).get("results") or []
        if not results:
            self._not_found(place)
        covered = [r for r in results if r.get("country_code") in COVERED_COUNTRIES]
        if not covered:
            error = OutsideCoverage(f"{place!r} resolved outside GB/IE")
            error.user_message = (
                f"I only know stargazing spots in the UK and Ireland, and "
                f"{cleaned} looks to be somewhere else."
            )
            raise error
        hit = covered[0]
        parts = [hit.get("name"), hit.get("admin1")]
        name = ", ".join(p for p in parts if p)
        zone_name = hit.get("timezone") or COVERED_COUNTRIES[hit["country_code"]]
        if zone_name not in COVERED_COUNTRIES.values():
            zone_name = COVERED_COUNTRIES[hit["country_code"]]
        irish = hit["country_code"] == "IE" or hit.get("admin1") == "Northern Ireland"
        return float(hit["latitude"]), float(hit["longitude"]), name, zone_name, irish

    def _nearby_reserve(self, lat: float, lon: float) -> dict | None:
        """A named nature reserve within about 15 km, or None.

        A backup, so any failure at all just leaves it out. Prefers the one
        farthest from the town centre (light thins out as you leave a town),
        but at least 2 km out and no more than 10 km when there is a choice.
        `handle` drops it if it is not closer than the recommended spot.
        """
        query = urlencode({
            "q": "nature reserve",
            "format": "jsonv2",
            "limit": 10,
            "bounded": 1,
            "viewbox": f"{lon - PARK_BOX_DEG_LON},{lat + PARK_BOX_DEG_LAT},"
                       f"{lon + PARK_BOX_DEG_LON},{lat - PARK_BOX_DEG_LAT}",
        })
        try:
            found = json.loads(self._fetch(NOMINATIM_URL + query, timeout=3.0))
        except Exception:
            return None
        reserves = []
        for item in found if isinstance(found, list) else []:
            try:
                name = (item.get("name") or "").strip()
                plat, plon = float(item["lat"]), float(item["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            if name:
                reserves.append({"name": name, "lat": plat, "lon": plon,
                                 "km": distance_km(lat, lon, plat, plon)})
        if not reserves:
            return None
        outside = [r for r in reserves if 2.0 <= r["km"] <= 10.0]
        pool = outside or reserves
        return max(pool, key=lambda r: r["km"]) if outside else min(pool, key=lambda r: r["km"])

    def _cloud_forecasts(self, spots: list[tuple[float, float]]) -> list[list[tuple[datetime, int]]]:
        """Hourly cloud cover (UTC hour, percent) for each spot, in order."""
        query = urlencode({
            "latitude": ",".join(f"{a:.3f}" for a, _ in spots),
            "longitude": ",".join(f"{b:.3f}" for _, b in spots),
            "hourly": "cloud_cover",
            "timezone": "GMT",
            "forecast_days": 2,
        })
        raw = self._get(FORECAST_URL + query, 6.0)
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                payload = [payload]
            series = []
            for entry in payload:
                hourly = entry["hourly"]
                pairs = [
                    (datetime.fromisoformat(t).replace(tzinfo=timezone.utc), int(c))
                    for t, c in zip(hourly["time"], hourly["cloud_cover"])
                    if c is not None
                ]
                if not pairs:
                    raise ValueError("empty series")
                series.append(pairs)
        except (ValueError, KeyError, TypeError) as exc:
            raise ServiceUnavailableError(f"forecast response unusable: {exc}")
        if len(series) != len(spots):
            raise ServiceUnavailableError(
                f"forecast returned {len(series)} series for {len(spots)} spots")
        return series

    def _get(self, url: str, timeout: float, not_found: str | None = None) -> bytes:
        """Fetch, turning network failures into errors a person can read."""
        try:
            return self._fetch(url, timeout=timeout)
        except urllib.error.HTTPError as exc:   # must come first: it IS a URLError
            if exc.code == 404 and not_found is not None:
                self._not_found(not_found)
            if exc.code == 429:
                retry = exc.headers.get("Retry-After") if exc.headers else None
                raise RateLimitedError(
                    f"rate limited ({exc.code})",
                    retry_after_seconds=int(retry) if retry and retry.isdigit() else None,
                )
            raise ServiceUnavailableError(f"upstream returned {exc.code}")
        except TimeoutError as exc:             # not a URLError: its own clause
            raise ServiceUnavailableError(f"timed out: {exc}")
        except urllib.error.URLError as exc:
            raise ServiceUnavailableError(f"could not reach upstream: {exc.reason}")

    @staticmethod
    def _not_found(place: str) -> None:
        error = PlaceNotFound(f"no match for {place!r}")
        error.user_message = f"I could not find {place} in the UK or Ireland."
        raise error

    @staticmethod
    def _sentence(data: dict, widened: bool, max_km) -> str:
        spot = (f"{data['best_spot']}, {data['best_spot_distance_km']:g} km away, "
                f"has {data['best_spot_cloud_percent']}% cloud, clearest around "
                f"{data['best_time']}")
        verdict = ("a good night to look up" if data["good_night"]
                   else "cloud may spoil it, even at the best spot")
        moon = data["moon_illumination_percent"]
        if moon >= 70:
            moon_note = f"a bright {data['moon_phase']} ({moon}% lit) will wash out faint stars"
        elif moon <= 25:
            moon_note = f"the moon is a {data['moon_phase']} ({moon}% lit), so the sky stays dark"
        else:
            moon_note = f"the moon is a {data['moon_phase']} ({moon}% lit)"
        dark = ("properly dark" if data["fully_dark"] else "as dark as it gets")
        sentence = (f"{spot}; it is {dark} from {data['dark_from']} to "
                    f"{data['dark_until']}, {moon_note}: {verdict}.")
        if widened:
            sentence = (f"Nothing on my list is within {max_km} km, so this is the "
                        f"nearest: {sentence}")
        return sentence
