import os
import re
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

from tools.flight_tool import search_flights, build_trip_legs, DEFAULT_ORIGIN_IATA, iata_to_city_name
from tools.duffel_client import get_offers_structured
from tools.tavily_tool import tavily_search_structured

import json
from collections import defaultdict


def get_database_url():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set in the environment variables.")

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)


# =========================
# State
# =========================
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_offers: list      # structured, from Duffel
    live_status: str         # AviationStack — still fine as text, it's just informational
    hotel_results: list      # structured, from Tavily
    itinerary: str
    llm_calls: int
    budget_breakdown: dict


# =========================
# Flight Agent
# =========================
def flight_agent(state: TravelState):
    query = state["user_query"]
    legs = build_trip_legs(query, trip_length_days=7)

    all_offers = []
    for leg in legs:
        offers = get_offers_structured(leg["dep"], leg["arr"], leg["date"])
        for o in offers:
            o["leg_label"] = f"{leg['dep']} -> {leg['arr']} on {leg['date']}"
        all_offers.extend(offers)

    live_status = search_flights(query)  # unchanged, purely informational text

    return {
        "flight_offers": all_offers,
        "live_status": live_status,
        "messages": [AIMessage(content="Flight search results have been retrieved.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Hotel Agent
# =========================
def hotel_agent(state: TravelState):
    query = state["user_query"]
    legs = build_trip_legs(query, trip_length_days=7)
    cities = list(dict.fromkeys([leg["arr"] for leg in legs])) or []

    hotel_domains = ["booking.com", "agoda.com", "tripadvisor.com", "hotels.com"]
    all_hotels = []

    if not cities:
        results = tavily_search_structured(f"Best budget hotels for {query}", include_domains=hotel_domains)
        for r in results:
            r["city"] = None
        all_hotels.extend(results)
    else:
        for city_iata in cities:
            city_name = iata_to_city_name(city_iata)  # "BOM" -> "Mumbai", not the code
            results = tavily_search_structured(f"Best budget hotels to stay in {city_name}", include_domains=hotel_domains)
            for r in results:
                r["city"] = city_iata
            all_hotels.extend(results)

    return {
        "hotel_results": all_hotels,
        "messages": [AIMessage(content="Hotel search results have been retrieved.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

def _format_offers_for_prompt(offers: list) -> str:
    """Text formatting happens ONCE, right before the LLM call — not scattered across tools."""
    if not offers:
        return "No bookable fare offers were found for this route/date."
    lines = []
    for o in offers:
        price = f"{o['price_amount']} {o['price_currency']}" if o["price_amount"] else "Price unavailable"
        if o.get("price_converted"):
            price += f" (~{o['price_converted']:,.2f} {o['target_currency']})"
        seg_text = "; ".join(f"{s['dep']}->{s['arr']} {s['flight_no']} dep {s['dep_time']} arr {s['arr_time']}" for s in o["segments"])
        lines.append(f"[{o.get('leg_label','')}] {o['airline']} — {price} — {seg_text}")
    return "\n".join(lines)


def _format_hotels_for_prompt(hotels: list) -> str:
    if not hotels:
        return "No hotel results found."
    # Show priced hotels first so the LLM prefers real numbers over guessing
    hotels_sorted = sorted(hotels, key=lambda h: h.get("price_inr") is None)
    lines = []
    for h in hotels_sorted:
        city_tag = f"[{h['city']}] " if h.get("city") else ""
        if h.get("price_inr"):
            price_tag = f" | ~₹{h['price_inr']['amount_inr']:,.0f}/night (converted from {h['price_inr']['source_currency']})"
        else:
            price_tag = " | price not listed"
        lines.append(f"{city_tag}{h['title']}{price_tag} — {h['snippet']} ({h['url']})")
    return "\n".join(lines)

from datetime import datetime
 
 
def _extract_trip_days(user_query: str, flight_offers: list | None = None) -> int:
    """
    Figures out how many days the trip should span, in priority order:
    1. An explicit "N day(s)" / "N-day" mention in the user's query.
    2. The gap between the earliest and latest flight-leg departure dates,
       if flight_offers has enough data (covers cases where the query
       doesn't state a duration but does give travel dates).
    3. A conservative default of 7.
    """
    match = re.search(r"\b(\d{1,2})\s*[- ]?days?\b", user_query, re.IGNORECASE)
    if match:
        days = int(match.group(1))
        if 1 <= days <= 60:
            return days
 
    if flight_offers:
        try:
            dep_dates = sorted(
                {
                    o["segments"][0]["dep_time"][:10]
                    for o in flight_offers
                    if o.get("segments") and o["segments"][0].get("dep_time")
                }
            )
            if len(dep_dates) >= 2:
                span = (
                    datetime.strptime(dep_dates[-1], "%Y-%m-%d")
                    - datetime.strptime(dep_dates[0], "%Y-%m-%d")
                ).days
                if 1 <= span <= 60:
                    return span
        except (KeyError, ValueError, IndexError, TypeError):
            pass
 
    return 7  # conservative fallback — matches the most common query pattern
 
 
def _trim_itinerary_to_days(itinerary_text: str, trip_days: int) -> str:
    """
    Safety net: even with the constrained prompt below, keep only the
    first `trip_days` "### Day N" blocks so an over-generating model can
    never surface a longer trip than the user asked for.
    """
    blocks = re.split(r"(?=### Day \d+:)", itinerary_text)
    day_blocks = [b for b in blocks if b.strip().startswith("### Day")]
    if len(day_blocks) <= trip_days:
        return itinerary_text
 
    preamble = blocks[0] if blocks and not blocks[0].strip().startswith("### Day") else ""
    return preamble + "".join(day_blocks[:trip_days])


# =========================
# Itinerary Agent
# =========================
def itinerary_agent(state: TravelState):
    flight_text = _format_offers_for_prompt(state["flight_offers"])
    hotel_text = _format_hotels_for_prompt(state["hotel_results"])
 
    trip_days = _extract_trip_days(state["user_query"], state.get("flight_offers"))
 
    prompt = f"""
Create a complete, detailed DAY-BY-DAY travel itinerary for this trip.
 
User Query: {state['user_query']}
 
This trip is EXACTLY {trip_days} days long — from the day the traveler departs
their home city to the day they arrive back. Produce EXACTLY {trip_days} day
entries, Day 1 through Day {trip_days}, and nothing more. Once the traveler
returns home, the itinerary ends immediately — do NOT add extra days exploring
the home city after the return flight.
 
Flight Offers:
{flight_text}
 
Hotel Options:
{hotel_text}
 
STRICT FORMAT REQUIREMENTS:
- Cover every day from Day 1 to Day {trip_days} individually. Do NOT group days
  together (e.g. never write "Day 2-6" or "Day 1-7" — write Day 1, Day 2, Day 3,
  ... separately, even if the destination city doesn't change).
- Do NOT produce more than {trip_days} day entries under any circumstances.
- Each day MUST use this exact header format: "### Day N: <date> - <short theme>"
- Each day must list AT LEAST 3 specific, named activities or attractions
  (real landmark/museum/neighborhood names for the actual destination city —
  not generic filler like "explore the city" or "enjoy local cuisine").
- Include a rough time-of-day structure per day (Morning / Afternoon / Evening)
  where it makes sense.
- On travel days, include the flight details AND at least one activity once
  the traveler has settled in (unless arrival is very late).
- Make it practical and budget-aware, using the flight and hotel data above.
"""
 
    response = llm.invoke([
        SystemMessage(content=(
            "You are an expert travel assistant who writes thorough, non-repetitive, "
            "day-by-day itineraries. You never merge multiple days into one entry, and "
            "you never exceed the exact trip length you are given."
        )),
        HumanMessage(content=prompt)
    ])
 
    itinerary_text = _trim_itinerary_to_days(response.content, trip_days)
 
    return {
        "itinerary": itinerary_text,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def _cheapest_per_leg(offers: list):
    groups = defaultdict(list)
    for o in offers:
        groups[o.get("leg_label", "Unknown leg")].append(o)
    result = {}
    for leg, offs in groups.items():
        offs_sorted = sorted(offs, key=lambda o: o.get("price_amount") or float("inf"))
        if offs_sorted:
            result[leg] = offs_sorted[0]
    return result


def _compute_core_budget(flight_offers: list, hotel_results: list, nights: int = 6):
    """Deterministic budget for the parts we have real data for."""
    budget = {}

    cheapest = _cheapest_per_leg(flight_offers)
    flight_total = sum(
        o.get("price_converted") or 0 for o in cheapest.values() if o.get("price_converted")
    )
    if flight_total:
        budget["Flights"] = round(flight_total, 2)

    priced_hotels = [h for h in hotel_results if h.get("price_inr")]
    if priced_hotels:
        avg_nightly = sum(h["price_inr"]["amount_inr"] for h in priced_hotels) / len(priced_hotels)
        budget["Accommodation"] = round(avg_nightly * nights, 2)

    return budget


def budget_agent(state: TravelState):
    query = state["user_query"]
    core_budget = _compute_core_budget(state["flight_offers"], state["hotel_results"])

    prompt = f"""
You are estimating REALISTIC supplementary trip costs in INR for this trip.

User Query: {query}

Already-known costs (do NOT re-estimate these, they're handled separately):
{json.dumps(core_budget, indent=2)}

Itinerary:
{state['itinerary']}

Estimate ONLY these additional categories, in INR, for the ENTIRE trip (not per day):
- "Local Transportation" (taxis, metro, trains between/within cities)
- "Food & Dining"
- "Sightseeing & Activities" (entry tickets, tours, passes)
- "Travel Insurance"
- "Miscellaneous" (SIM card, tips, shopping buffer, unexpected costs)

Base estimates on the destination's typical cost of living, trip length, and
number of travelers implied by the query (assume 1 traveler if not specified).

Respond with ONLY a valid JSON object — no markdown, no code fences, no explanation.
Example: {{"Local Transportation": 8000, "Food & Dining": 25000, "Sightseeing & Activities": 15000, "Travel Insurance": 3000, "Miscellaneous": 5000}}
"""

    response = llm.invoke([
        SystemMessage(content="You are a meticulous travel budget analyst. You always respond with valid JSON only, nothing else."),
        HumanMessage(content=prompt)
    ])

    raw = re.sub(r"^```(?:json)?|```$", "", response.content.strip(), flags=re.MULTILINE).strip()

    misc_budget = {}
    try:
        parsed = json.loads(raw)
        for k, v in parsed.items():
            try:
                misc_budget[k] = round(float(v), 2)
            except (TypeError, ValueError):
                continue
    except json.JSONDecodeError:
        misc_budget = {}  # graceful fallback — core budget still works

    combined_budget = {**core_budget, **misc_budget}

    return {
        "budget_breakdown": combined_budget,
        "messages": [AIMessage(content="Budget breakdown estimated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }




def final_agent(state: TravelState):
    flight_text = _format_offers_for_prompt(state["flight_offers"])
    hotel_text = _format_hotels_for_prompt(state["hotel_results"])
    budget = state.get("budget_breakdown", {})
    budget_text = "\n".join(f"- {k}: ₹{v:,.2f}" for k, v in budget.items()) or "No structured budget available."
    total = sum(budget.values())


    final_prompt = f"""
Generate the final travel response for: {state['user_query']}

Flight Offers (use exactly, do not invent):
{flight_text}

Hotel Options (use exactly, do not invent):
{hotel_text}

Full Day-by-Day Itinerary (already finalized — reproduce it IN FULL, day by day, do not summarize):
{state['itinerary']}

Structured Budget Breakdown (use these EXACT figures — do not recalculate or invent new numbers):
{budget_text}
Total: ₹{total:,.2f}

Format as:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

CRITICAL RULES for section 4 (Day-by-Day Itinerary):
- Reproduce the itinerary above AS-IS, one entry per day, same headers, all detail intact.
- Never collapse multiple days into a range like "Day 1-7".

CRITICAL RULES for section 5 (Estimated Budget):
- List each category and amount exactly as given above, in INR.
- State the total exactly as given above.
- Compare the total against the user's stated budget (if any) and say clearly
  whether the trip fits within budget or exceeds it, and by how much.

Other rules:
- Use only the data provided above. If a field is unavailable, say so explicitly.
- Do not fabricate prices, hotel names, or flight numbers.
"""

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant. You preserve the itinerary and budget figures you're given exactly, without altering them."),
        HumanMessage(content=final_prompt)
    ])

    return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}




# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("budget_agent", budget_agent) 
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "budget_agent")  # <-- new
graph.add_edge("budget_agent", "final_agent")
graph.add_edge("final_agent", END)

# =========================
# PostgreSQL Checkpointer
# =========================
DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# Function for FastAPI
# =========================
def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_offers": [],
            "live_status": "",
            "hotel_results": [],
            "itinerary": "",
            "budget_breakdown": {},
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_offers": result.get("flight_offers", []),
        "live_status": result.get("live_status", ""),
        "hotel_results": result.get("hotel_results", []),
        "itinerary": result.get("itinerary", ""),
        "budget_breakdown": result.get("budget_breakdown", {}),
        "llm_calls": result.get("llm_calls", 0),
    }