import os
import re
import pycountry
import certifi
import airportsdata
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

DEFAULT_ORIGIN_IATA = os.getenv("DEFAULT_ORIGIN_IATA", "BOM")

BASE_URL = "http://api.aviationstack.com/v1/flights"

AIRPORTS = airportsdata.load('IATA')  # Load airport data with IATA codes

COUNTRY_ALIASES = {
    "usa": "US",
    "u.s.a": "US",
    "u.s.": "US",
    "america": "US",
    "united states": "US",
    "uk": "GB",
    "u.k.": "GB",
    "britain": "GB",
    "england": "GB",
    "uae": "AE",
    "dubai": "AE",
    "south korea": "KR",
    "korea": "KR",
    "russia": "RU",
    "vietnam": "VN",
    "bangladesh": "BD",
    "india": "IN",
    "japan": "JP",
    "china": "CN",
    "singapore": "SG",
    "malaysia": "MY",
    "thailand": "TH",
    "indonesia": "ID",
    "nepal": "NP",
    "qatar": "QA",
    "saudi arabia": "SA",
    "turkey": "TR",
    "canada": "CA",
    "australia": "AU",
    "germany": "DE",
    "france": "FR",
    "italy": "IT",
    "spain": "ES",
}

# Preferred main airport for country-level search
COUNTRY_MAIN_AIRPORT = {
    "BD": "DAC",
    "IN": "BOM",
    "JP": "NRT",
    "US": "JFK",
    "GB": "LHR",
    "AE": "DXB",
    "SG": "SIN",
    "MY": "KUL",
    "TH": "BKK",
    "ID": "CGK",
    "CN": "PEK",
    "KR": "ICN",
    "NP": "KTM",
    "QA": "DOH",
    "SA": "JED",
    "TR": "IST",
    "CA": "YYZ",
    "AU": "SYD",
    "DE": "FRA",
    "FR": "CDG",
    "IT": "FCO",
    "ES": "MAD",
}

CITY_MAIN_AIRPORT = {
    "dhaka": "DAC",
    "delhi": "DEL",
    "new delhi": "DEL",
    "mumbai": "BOM",
    "kolkata": "CCU",
    "chennai": "MAA",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "tokyo": "NRT",
    "osaka": "KIX",
    "kyoto": "KIX",
    "new york": "JFK",
    "london": "LHR",
    "dubai": "DXB",
    "singapore": "SIN",
    "kuala lumpur": "KUL",
    "bangkok": "BKK",
    "doha": "DOH",
    "istanbul": "IST",
    "toronto": "YYZ",
    "sydney": "SYD",
    "paris": "CDG",
    "rome": "FCO",
    "madrid": "MAD",
    "frankfurt": "FRA",
    "las vegas": "LAS",
    "los angeles": "LAX",
    "san francisco": "SFO",
    "chicago": "ORD",
    "miami": "MIA",
    "boston": "BOS",
    "washington": "IAD",
}


def clean_text(text: str) -> str:
    """
    Cleans the input text by removing special characters and extra spaces.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    stop_words = [
        "flight", "flights", "ticket", "tickets", "trip", "travel",
        "plan", "complete", "days", "day", "including", "hotel",
        "hotels", "sightseeing", "under", "budget", "info", "information"
    ]
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words).strip()


def country_name_to_code(text: str) -> str:
    """
    Converts a country name to its corresponding ISO 3166-1 alpha-2 code.
    If the country name is not found, returns None.
    """
    text = clean_text(text)

    if text in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[text]

    try:
        country = pycountry.countries.lookup(text)
        return country.alpha_2
    except LookupError:
        pass

    # Detect country name inside longer text
    for country in pycountry.countries:
        country_name = country.name.lower()
        if country_name in text:
            return country.alpha_2

    for alias, code in COUNTRY_ALIASES.items():
        if alias in text:
            return code

    return None


def airport_country_matches(airport: dict, country_code: str) -> bool:
    airport_country = str(airport.get("country", "")).upper().strip()

    if airport_country == country_code:
        return True

    try:
        country = pycountry.countries.get(alpha_2=country_code)
        if country and airport_country.lower() == country.name.lower():
            return True
    except Exception:
        pass

    return False


def get_best_airport_for_country(country_code: str) -> str:
    """
    Returns the best airport IATA code for a given country code.
    If no suitable airport is found, returns None.
    """

    preferred = COUNTRY_MAIN_AIRPORT.get(country_code)

    if preferred and preferred in AIRPORTS:
        return preferred

    candidates = []

    for iata, airport in AIRPORTS.items():
        if not iata:
            continue

        if airport_country_matches(airport, country_code):
            name = str(airport.get("name", "")).lower()
            city = str(airport.get("city", "")).lower()

            score = 0

            if "international" in name:
                score += 50
            if "intl" in name:
                score += 40
            if "capital" in name:
                score += 20
            if city:
                score += 5

            candidates.append((score, iata))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]  # Return the IATA code of the best airport


def resolve_location_to_iata(location: str) -> str:
    """
    Resolves a location (city or country) to its corresponding IATA airport code.
    If the location is not found, returns None.

    Order matters here:
      1. Known city aliases (fast, exact)
      2. Known country aliases (fast, exact) -> best airport for that country
      3. Literal 3-letter IATA code (ONLY after alias checks, so a 3-letter
         word like "usa" is never mistaken for a literal airport code)
      4. Fuzzy match against airport names/cities in the full database
    """

    if not location:
        return None

    raw_location = location.strip()
    location_clean = clean_text(raw_location)

    if not location_clean:
        return None

    # 1. City preferred airport check
    if location_clean in CITY_MAIN_AIRPORT:
        return CITY_MAIN_AIRPORT[location_clean]

    # 2. Country preferred airport check
    country_code = country_name_to_code(location_clean)
    if country_code:
        return get_best_airport_for_country(country_code)

    # 3. Direct IATA code check (only after alias checks are exhausted)
    if re.fullmatch(r"[A-Za-z]{3}", raw_location):
        code = raw_location.upper()
        if code in AIRPORTS:
            return code

    # 4. Fuzzy match in airport names
    city_matches = []
    for iata, airport in AIRPORTS.items():
        if not iata:
            continue

        name = str(airport.get("name", "")).lower().strip()
        city = str(airport.get("city", "")).lower().strip()

        score = 0

        if city == location_clean:
            score += 100
        elif location_clean in city:
            score += 70

        if location_clean in name:
            score += 50

        if "international" in name:
            score += 10

        if score > 0:
            city_matches.append((score, iata, airport))

    if city_matches:
        city_matches.sort(reverse=True)
        return city_matches[0][1]  # Return the IATA code of the best matching airport

    return None

def iata_to_city_name(iata: str) -> str:
    """Reverse-lookup: IATA code -> human city name, for building search queries."""
    airport = AIRPORTS.get(iata)
    if airport:
        return airport.get("city") or airport.get("name") or iata
    return iata

def find_location_mentions(query: str) -> list[str]:
    """
    Finds country/city mentions in the SAME ORDER
    they appear in the user's sentence.
    """

    q = query.lower()

    candidates = set()

    # Country aliases
    candidates.update(COUNTRY_ALIASES.keys())

    # Official country names
    for country in pycountry.countries:
        candidates.add(country.name.lower())

    # Cities
    candidates.update(CITY_MAIN_AIRPORT.keys())

    found = []

    for candidate in candidates:
        m = re.search(rf"\b{re.escape(candidate)}\b", q)
        if m:
            found.append((m.start(), candidate))

    # Sort by position in the sentence
    found.sort(key=lambda x: x[0])

    mentions = []
    seen = set()

    for _, loc in found:
        if loc not in seen:
            mentions.append(loc)
            seen.add(loc)

    return mentions


def build_trip_legs(query: str, trip_length_days: int = 7, default_origin_iata: str = None):
    """
    Returns a list of dicts: [{"dep": "BOM", "arr": "JFK", "date": "2026-08-20"}, ...]
    Advances the date for each subsequent leg so a return leg isn't
    dated the same day as the outbound.
    """
    default_origin_iata = default_origin_iata or DEFAULT_ORIGIN_IATA
    mentions = find_location_mentions(query)
    if not mentions:
        return []

    q_lower = query.lower()
    origin = None
    m = re.search(
        r"from\s+([a-zA-Z\s]+?)(?:$|[,.!?]|\s+(?:to|and|via|including|covering|visiting))",
        q_lower,
    )
    if m:
        candidate = clean_text(m.group(1))
        for mention in mentions:
            if clean_text(mention) == candidate or candidate in clean_text(mention):
                origin = mention
                break
    if origin is None:
        origin = mentions[0]

    destinations = [d for d in mentions if clean_text(d) != clean_text(origin)]

    dep_iata = resolve_location_to_iata(origin) or default_origin_iata

    # Extract a start date if present, else 14 days out
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)
    start_date = (
        datetime.strptime(date_match.group(1), "%Y-%m-%d")
        if date_match else datetime.now() + timedelta(days=14)
    )

    legs = []
    prev_iata = dep_iata
    current_date = start_date
    resolved_dests = []
    for d in destinations:
        arr = resolve_location_to_iata(d)
        if arr and arr != prev_iata and arr not in resolved_dests:
            legs.append({"dep": prev_iata, "arr": arr, "date": current_date.strftime("%Y-%m-%d")})
            prev_iata = arr
            resolved_dests.append(arr)
            current_date += timedelta(days=max(1, trip_length_days // max(1, len(destinations) + 1)))

    if prev_iata != dep_iata:
        return_date = start_date + timedelta(days=trip_length_days)
        legs.append({"dep": prev_iata, "arr": dep_iata, "date": return_date.strftime("%Y-%m-%d")})

    return legs


def parse_route(query: str):
    """
    Returns:
        (dep_iata, arr_iata)

    Examples:
        "Flights from India to Japan"
            -> ("BOM", "NRT")

        "Japan trip from India"
            -> ("BOM", "NRT")

        "Travel from Mumbai"
            -> ("BOM", None)

        "Flights to Tokyo"
            -> (DEFAULT_ORIGIN_IATA, "HND"/"NRT")

        "BOM to NRT"
            -> ("BOM", "NRT")

        "Global flights"
            -> (None, None)
    """

    q = query.strip()
    q_lower = q.lower()

    # --------------------------------------------------
    # Global queries
    # --------------------------------------------------
    global_keywords = [
        "all country",
        "all countries",
        "global flight",
        "global flights",
        "all flight",
        "all flights",
        "worldwide flight",
        "worldwide flights",
    ]

    if any(k in q_lower for k in global_keywords):
        return None, None

    # --------------------------------------------------
    # Direct IATA codes
    # Example: BOM to NRT
    # --------------------------------------------------
    direct_code_match = re.fullmatch(r"\s*([A-Za-z]{3})\s*(?:to|->)\s*([A-Za-z]{3})\s*", q)
    if direct_code_match:
        c1, c2 = direct_code_match.group(1).upper(), direct_code_match.group(2).upper()
        if c1 in AIRPORTS and c2 in AIRPORTS:
            return c1, c2

    # --------------------------------------------------
    # Find all location mentions
    # --------------------------------------------------
    mentions = find_location_mentions(query)

    origin = None
    destination = None

    # --------------------------------------------------
    # Pattern: from X to Y
    # --------------------------------------------------
    match = re.search(
        r"from\s+(.+?)\s+to\s+(.+?)(?:$|[,.!?]|(?:\s+(?:including|with|under|for|budget|trip|days?|hotels?|flights?|sightseeing)))",
        q_lower,
    )

    if match:
        origin = match.group(1).strip()
        destination = match.group(2).strip()

    # --------------------------------------------------
    # Pattern: to Y from X
    # --------------------------------------------------
    if origin is None and destination is None:

        match = re.search(
            r"to\s+(.+?)\s+from\s+(.+?)(?:$|[,.!?]|(?:\s+(?:including|with|under|for|budget|trip|days?|hotels?|flights?|sightseeing)))",
            q_lower,
        )

        if match:
            destination = match.group(1).strip()
            origin = match.group(2).strip()

    # --------------------------------------------------
    # Pattern: from X
    # --------------------------------------------------
    if origin is None:

        match = re.search(
            r"from\s+(.+?)(?:$|[,.!?]|(?:\s+(?:including|with|under|for|budget|trip|days?|hotels?|flights?|sightseeing)))",
            q_lower,
        )

        if match:
            origin = match.group(1).strip()

    # --------------------------------------------------
    # Pattern: to Y
    # --------------------------------------------------
    if destination is None:

        match = re.search(
            r"to\s+(.+?)(?:$|[,.!?]|(?:\s+(?:including|with|under|for|budget|trip|days?|hotels?|flights?|sightseeing)))",
            q_lower,
        )

        if match:
            destination = match.group(1).strip()

    # --------------------------------------------------
    # Infer missing destination/origin from mentions
    # --------------------------------------------------

    if mentions:

        # resolve origin if regex captured it
        origin_clean = clean_text(origin) if origin else None
        destination_clean = clean_text(destination) if destination else None

        if origin is None:

            # if query contains "from", last mention is usually origin
            if "from" in q_lower:
                origin = mentions[-1]
            elif len(mentions) >= 2:
                origin = mentions[0]

        if destination is None:

            for m in mentions:
                if clean_text(m) != clean_text(origin):
                    destination = m
                    break

    dep_iata = resolve_location_to_iata(origin) if origin else None
    arr_iata = resolve_location_to_iata(destination) if destination else None

    # --------------------------------------------------
    # Default origin
    # --------------------------------------------------
    if dep_iata is None and arr_iata is not None:
        dep_iata = DEFAULT_ORIGIN_IATA

    return dep_iata, arr_iata


def format_flight(flight: dict):
    airline = flight.get("airline", {}).get("name") or "Unknown airline"
    flight_number = flight.get("flight", {}).get("iata") or "Unknown flight number"
    status = flight.get("flight_status") or "Unknown"

    dep = flight.get("departure", {}) or {}
    arr = flight.get("arrival", {}) or {}

    dep_airport = dep.get("airport") or "Unknown departure airport"
    dep_iata = dep.get("iata") or "Unknown"
    dep_terminal = dep.get("terminal") or "N/A"
    dep_gate = dep.get("gate") or "N/A"
    dep_scheduled = dep.get("scheduled") or "Unknown"
    dep_delay = dep.get("delay")
    dep_delay_text = f"{dep_delay} minutes" if dep_delay is not None else "N/A"

    arr_airport = arr.get("airport") or "Unknown arrival airport"
    arr_iata = arr.get("iata") or "Unknown"
    arr_terminal = arr.get("terminal") or "N/A"
    arr_gate = arr.get("gate") or "N/A"
    arr_scheduled = arr.get("scheduled") or "Unknown"
    arr_delay = arr.get("delay")
    arr_delay_text = f"{arr_delay} minutes" if arr_delay is not None else "N/A"

    return f"""
Airline: {airline}
Flight: {flight_number}
Status: {status}

Departure:
- Airport: {dep_airport}
- IATA: {dep_iata}
- Terminal: {dep_terminal}
- Gate: {dep_gate}
- Scheduled: {dep_scheduled}
- Delay: {dep_delay_text}

Arrival:
- Airport: {arr_airport}
- IATA: {arr_iata}
- Terminal: {arr_terminal}
- Gate: {arr_gate}
- Scheduled: {arr_scheduled}
- Delay: {arr_delay_text}
""".strip()


def search_flights(query: str, limit: int = 10):
    if not API_KEY:
        return (
            "Flight API error: AVIATIONSTACK_API_KEY is missing.\n"
            "Please add this in your .env file:\n"
            "AVIATIONSTACK_API_KEY=your_api_key_here"
        )

    dep_iata, arr_iata = parse_route(query)

    if dep_iata and dep_iata not in AIRPORTS:
        dep_iata = None
    if arr_iata and arr_iata not in AIRPORTS:
        arr_iata = None

    params = {
        "access_key": API_KEY,
        "limit": min(limit, 100),
    }

    if dep_iata:
        params["dep_iata"] = dep_iata

    if arr_iata:
        params["arr_iata"] = arr_iata

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"Flight API request failed: {e}"
    except ValueError:
        return "Flight API returned invalid JSON."

    if "error" in data:
        error = data["error"]
        return (
            "Flight API error:\n"
            f"Code: {error.get('code', 'Unknown')}\n"
            f"Message: {error.get('message', 'Unknown error')}"
        )

    flight_data = data.get("data", [])

    if not flight_data:
        route_text = ""

        if dep_iata and arr_iata:
            route_text = f" for route {dep_iata} to {arr_iata}"
        elif dep_iata:
            route_text = f" from {dep_iata}"
        elif arr_iata:
            route_text = f" to {arr_iata}"

        return (
            f"No live flight data found{route_text}.\n\n"
            "Note: AviationStack provides live/status flight data, not ticket prices. "
            "For actual fare prices, use a flight-pricing API such as Amadeus."
        )

    route_info = "Global live flights"

    if dep_iata and arr_iata:
        route_info = f"Live flights from {dep_iata} to {arr_iata}"
    elif dep_iata:
        route_info = f"Live flights from {dep_iata}"
    elif arr_iata:
        route_info = f"Live flights to {arr_iata}"

    formatted_flights = [format_flight(flight) for flight in flight_data[:limit]]

    return f"{route_info}\n\n" + "\n\n---\n\n".join(formatted_flights)


if __name__ == "__main__":
    print(resolve_location_to_iata("usa"))  # sanity check -> should print JFK
    print("\n" + "=" * 80 + "\n")
    print(search_flights("flights from India to Japan"))
    print("\n" + "=" * 80 + "\n")
    print(search_flights("all country flight info"))