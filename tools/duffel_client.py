"""
duffel_client.py

Adds Duffel API support for:
  - Ticket FARES (actual bookable prices)
  - FUTURE flight search (any date, not just "today")

Works alongside the existing AviationStack-based flight_finder.py,
which stays responsible for live status / delay / gate info.

Reuses location parsing from flight_finder.py so users can keep typing
natural queries like "flights from Mumbai to Tokyo on 2026-09-15".
"""

import os
import re
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Reuse your existing location + route parsing logic
from tools.flight_tool import (
    resolve_location_to_iata,
    find_location_mentions,
    clean_text,
    parse_route,
    DEFAULT_ORIGIN_IATA,
)

load_dotenv()

DUFFEL_API_KEY = os.getenv("DUFFEL_API_KEY")
DUFFEL_BASE_URL = "https://api.duffel.com/air/offer_requests"

DUFFEL_HEADERS = {
    "Authorization": f"Bearer {DUFFEL_API_KEY}",
    "Duffel-Version": "v2",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# --------------------------------------------------------------------
# Currency conversion (USD -> INR, or any base -> target)
# Uses frankfurter.app: free, no API key required.
# Rate is fetched once per run and cached, with a hardcoded fallback
# in case the network call fails (e.g. offline, rate limited).
# --------------------------------------------------------------------
FRANKFURTER_URL = "https://api.frankfurter.app/latest"
FALLBACK_USD_TO_INR = 87.5  # rough fallback if the live rate can't be fetched

_rate_cache = {}


def get_exchange_rate(base: str = "USD", target: str = "INR") -> float:
    """
    Returns how many `target` units equal 1 `base` unit (e.g. 1 USD -> X INR).
    Cached per (base, target) pair for the life of the process.
    """
    key = (base.upper(), target.upper())
    if key in _rate_cache:
        return _rate_cache[key]

    try:
        response = requests.get(
            FRANKFURTER_URL,
            params={"from": base.upper(), "to": target.upper()},
            timeout=10,
        )
        data = response.json()
        rate = data.get("rates", {}).get(target.upper())
        if rate is None:
            raise ValueError("Rate missing in response")
    except (requests.exceptions.RequestException, ValueError):
        rate = FALLBACK_USD_TO_INR if key == ("USD", "INR") else None

    _rate_cache[key] = rate
    return rate


def convert_amount(amount: float, base: str = "USD", target: str = "INR") -> float:
    rate = get_exchange_rate(base, target)
    if rate is None:
        return None
    return round(amount * rate, 2)


# --------------------------------------------------------------------
# Date parsing — pulls a travel date out of the query, defaults to
# "2 weeks from today" if the user didn't specify one (future search).
# --------------------------------------------------------------------
def parse_travel_date(query: str) -> str:
    """
    Looks for a YYYY-MM-DD date in the query. If not found, defaults
    to 14 days from today so future-search still works out of the box.
    """
    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)
    if match:
        return match.group(1)

    # crude support for "DD Month" or "Month DD" could be added here later
    default_date = datetime.now() + timedelta(days=14)
    return default_date.strftime("%Y-%m-%d")


# --------------------------------------------------------------------
# Route parsing reused from flight_finder's location logic, but kept
# separate/local here so this module doesn't depend on parse_route's
# "global keywords" and stop-word behavior tuned for AviationStack text.
# --------------------------------------------------------------------
def parse_fare_route(query: str):
    return parse_route(query)


# --------------------------------------------------------------------
# Core Duffel calls
# --------------------------------------------------------------------
def create_offer_request(dep_iata: str, arr_iata: str, depart_date: str,
                          passengers: int = 1, cabin_class: str = "economy"):
    """
    Creates a Duffel offer_request, which triggers a live search across
    connected airlines and returns priced offers.
    """
    if not DUFFEL_API_KEY:
        return None, "Duffel API error: DUFFEL_API_KEY is missing in .env"

    payload = {
        "data": {
            "slices": [
                {
                    "origin": dep_iata,
                    "destination": arr_iata,
                    "departure_date": depart_date,
                }
            ],
            "passengers": [{"type": "adult"} for _ in range(passengers)],
            "cabin_class": cabin_class,
        }
    }

    try:
        response = requests.post(
            DUFFEL_BASE_URL,
            headers=DUFFEL_HEADERS,
            params={"return_offers": "true"},
            json=payload,
            timeout=30,
        )
        data = response.json()
    except requests.exceptions.RequestException as e:
        return None, f"Duffel API request failed: {e}"
    except ValueError:
        return None, "Duffel API returned invalid JSON."

    if response.status_code >= 400:
        errors = data.get("errors", [{"message": "Unknown error"}])
        msg = "; ".join(e.get("message", "Unknown error") for e in errors)
        return None, f"Duffel API error ({response.status_code}): {msg}"

    return data, None


def format_offer(offer: dict, target_currency: str = "INR") -> str:
    airline = offer.get("owner", {}).get("name", "Unknown airline")
    total_amount = offer.get("total_amount", "N/A")
    total_currency = offer.get("total_currency", "")
    expires_at = offer.get("expires_at", "N/A")

    # Convert price to the target currency (defaults to INR) if it isn't already
    price_line = f"Price: {total_amount} {total_currency}"
    if total_currency and total_currency.upper() != target_currency.upper():
        try:
            converted = convert_amount(float(total_amount), total_currency, target_currency)
            if converted is not None:
                price_line += f"  (~{converted:,.2f} {target_currency.upper()})"
            else:
                price_line += f"  (INR conversion unavailable)"
        except (TypeError, ValueError):
            pass

    slices_text = []
    for sl in offer.get("slices", []):
        for seg in sl.get("segments", []):
            dep = seg.get("origin", {}).get("iata_code", "N/A")
            arr = seg.get("destination", {}).get("iata_code", "N/A")
            dep_time = seg.get("departing_at", "N/A")
            arr_time = seg.get("arriving_at", "N/A")
            flight_no = f"{seg.get('marketing_carrier', {}).get('iata_code', '')}{seg.get('marketing_carrier_flight_number', '')}"
            slices_text.append(
                f"  {dep} -> {arr} | Flight {flight_no} | Dep: {dep_time} | Arr: {arr_time}"
            )

    return (
        f"Airline: {airline}\n"
        f"{price_line}\n"
        f"Offer expires: {expires_at}\n"
        "Itinerary:\n" + "\n".join(slices_text)
    )

def get_offers_structured(dep_iata, arr_iata, depart_date, passengers=1, limit=5, target_currency="INR"):
    """
    Returns a list of plain dicts, NOT formatted text. e.g.:
    [{
        "airline": "SriLankan Airlines",
        "price_amount": 95.26, "price_currency": "USD",
        "price_converted": 9061.13, "target_currency": "INR",
        "expires_at": "...",
        "segments": [{"dep": "BOM", "arr": "CMB", "flight_no": "UL0142",
                       "dep_time": "...", "arr_time": "..."}],
        "is_test_offer": False,
    }, ...]
    """
    if not dep_iata or not arr_iata:
        return []

    data, error = create_offer_request(dep_iata, arr_iata, depart_date, passengers)
    if error or not data:
        return []

    offers = data.get("data", {}).get("offers", [])
    # Drop sandbox synthetic offers up front — not just at display time
    offers = [o for o in offers if o.get("owner", {}).get("name", "").lower() != "duffel airways"]
    offers = sorted(offers, key=lambda o: float(o.get("total_amount", "999999")))

    structured = []
    for offer in offers[:limit]:
        total_amount = offer.get("total_amount")
        total_currency = offer.get("total_currency", "")
        converted = None
        if total_amount and total_currency and total_currency.upper() != target_currency.upper():
            try:
                converted = convert_amount(float(total_amount), total_currency, target_currency)
            except (TypeError, ValueError):
                pass

        segments = []
        for sl in offer.get("slices", []):
            for seg in sl.get("segments", []):
                segments.append({
                    "dep": seg.get("origin", {}).get("iata_code", "N/A"),
                    "arr": seg.get("destination", {}).get("iata_code", "N/A"),
                    "flight_no": f"{seg.get('marketing_carrier', {}).get('iata_code', '')}{seg.get('marketing_carrier_flight_number', '')}",
                    "dep_time": seg.get("departing_at", "N/A"),
                    "arr_time": seg.get("arriving_at", "N/A"),
                })

        structured.append({
            "airline": offer.get("owner", {}).get("name", "Unknown airline"),
            "price_amount": float(total_amount) if total_amount else None,
            "price_currency": total_currency,
            "price_converted": converted,
            "target_currency": target_currency,
            "expires_at": offer.get("expires_at", "N/A"),
            "segments": segments,
            "is_test_offer": False,
        })

    return structured

def search_fares(query: str, passengers: int = 1, limit: int = 5, target_currency: str = "INR"):
    """
    Main entry point: given a natural language query, returns formatted
    ticket fare offers (works for future dates too), with prices shown
    in both the original currency and `target_currency` (default INR).

    Example:
        search_fares("flights from Mumbai to Tokyo on 2026-09-15")
    """
    dep_iata, arr_iata = parse_fare_route(query)
    depart_date = parse_travel_date(query)

    if not dep_iata or not arr_iata:
        return (
            "Couldn't resolve both an origin and a destination for fare search.\n"
            f"Parsed origin: {dep_iata}, destination: {arr_iata}"
        )

    data, error = create_offer_request(dep_iata, arr_iata, depart_date, passengers)
    if error:
        return error

    offers = [o for o in offers if o.get("owner", {}).get("name", "").lower() != "duffel airways"]
    if not offers:
        return (
            f"No fare offers found for {dep_iata} -> {arr_iata} on {depart_date}.\n"
            "Try a different date or route."
        )

    # Duffel doesn't guarantee sort order, so sort by price ascending
    offers = sorted(offers, key=lambda o: float(o.get("total_amount", "999999")))

    formatted = [format_offer(o, target_currency) for o in offers[:limit]]
    header = f"Fares for {dep_iata} -> {arr_iata} on {depart_date} ({passengers} passenger(s))"
    return f"{header}\n\n" + "\n\n---\n\n".join(formatted)

def search_fares_leg(dep_iata, arr_iata, depart_date, passengers=1, limit=5, target_currency="INR"):
    if not dep_iata or not arr_iata:
        return f"Couldn't resolve this leg (origin={dep_iata}, destination={arr_iata})."

    data, error = create_offer_request(dep_iata, arr_iata, depart_date, passengers)
    if error:
        return f"{dep_iata} -> {arr_iata}: {error}"

    offers = data.get("data", {}).get("offers", [])
    if not offers:
        return f"No fare offers found for {dep_iata} -> {arr_iata} on {depart_date}."

    offers = sorted(offers, key=lambda o: float(o.get("total_amount", "999999")))
    formatted = [format_offer(o, target_currency) for o in offers[:limit]]
    return f"{dep_iata} -> {arr_iata} on {depart_date}\n\n" + "\n\n---\n\n".join(formatted)


if __name__ == "__main__":
    print(search_fares("flights from Mumbai to Tokyo on 2026-09-15"))