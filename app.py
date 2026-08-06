"""
TripMate AI — Streamlit UI
Frontend for the LangGraph-based multi-agent travel planner in backend.py
(flight_agent -> hotel_agent -> itinerary_agent -> final_agent)

This version consumes STRUCTURED data returned by run_travel_agent()
(flight_offers: list[dict], hotel_results: list[dict]) directly —
no regex parsing of formatted text.

Visual identity: an airport departure-board / boarding-pass motif —
navy + amber + a paper-ticket texture for flight and hotel cards.

Upgrades in this version:
  - Live staged progress while the agent pipeline runs (st.status), so the
    multi-agent architecture is visible instead of a single blank spinner.
    NOTE: backend.run_travel_agent() is currently a single blocking call,
    so the stage labels below are a best-effort sequence timed to the known
    pipeline order (flight -> hotel -> itinerary -> final), not a live feed
    from LangGraph. If/when backend exposes a streaming entry point (e.g.
    run_travel_agent_stream() yielding node updates), swap run_with_progress()
    to consume that instead of the timed loop — the st.status() call sites
    below don't need to change.
  - Actionable empty/error states instead of a single generic message.
  - Data-source badges (Duffel / Tavily) on result cards.
  - Itinerary rendered as day tabs instead of stacked expanders.
  - Budget tab is now interactive: nights/travelers sliders recompute the
    chart and total client-side, reusing the same helpers used server-side.
  - "How this was built" architecture panel (agent flow + run metrics).
  - Full result JSON export alongside the existing Markdown export.
  - Card entrance animation, hover elevation, and a mobile layout fix for
    the boarding-pass flight card (which previously assumed a wide viewport).
"""

import json
import threading
import time
import uuid
from collections import defaultdict

import streamlit as st
import plotly.graph_objects as go

from backend import run_travel_agent


def render_html(html: str) -> None:
    """
    st.markdown()'s parser treats 4+ space-indented lines that follow a
    blank line as a fenced code block. Our HTML snippets inherit Python's
    source indentation, and a conditional piece (e.g. an empty CHEAPEST
    badge) can leave a blank-looking line right before heavily-indented
    markup — which trips that rule and prints raw tags instead of
    rendering them. Dedenting every line before rendering avoids that.
    """
    lines = [line.strip() for line in html.strip("\n").split("\n")]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="TripMate AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Design tokens + styling
# ---------------------------------------------------------
# Palette: ink navy + departure-board amber + a signal teal,
# on a warm "paper ticket" surface. Mono face for anything that
# reads like flight-board data (codes, prices, times); a warm
# grotesque for everything else.
# =========================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --tm-navy: #0B2545;
            --tm-navy-2: #123a68;
            --tm-amber: #E8A33D;
            --tm-teal: #1F8A70;
            --tm-coral: #D2564F;
            --tm-paper: #FBF8F2;
            --tm-paper-2: #F3EEE2;
            --tm-ink: #1C2530;
            --tm-slate: #647188;
            --tm-line: #DED6C2;
            --tm-font-display: 'Manrope', sans-serif;
            --tm-font-mono: 'IBM Plex Mono', monospace;

            /*
             * This design was built against a light page. If the app is
             * deployed with Streamlit's dark theme, every native widget
             * (expanders, buttons, inputs, code chips) reads its colors
             * from these Streamlit theme variables — so we override the
             * variables themselves rather than patching each widget.
             */
            --background-color: var(--tm-paper);
            --secondary-background-color: var(--tm-paper-2);
            --text-color: var(--tm-ink);
            --primary-color: var(--tm-navy);
        }

        html, body, [class*="css"] { font-family: var(--tm-font-display); }

        /* Guard against horizontal scroll clipping content off the left
           edge (seen with the sidebar's resizable-width handle). */
        html, body { overflow-x: hidden; }
        .stApp {
            background: var(--tm-paper) !important;
            color: var(--tm-ink) !important;
            overflow-x: hidden;
        }

        /* --- Dark-theme text safety net ---------------------------------
         * The variables above don't reach Streamlit's own chrome (top
         * toolbar, buttons, alerts, plain markdown text, widget labels) —
         * those get their color set directly on inner elements by
         * Streamlit's dark theme, which beats inheritance from .stApp.
         * Each of those gets an explicit, !important override below
         * rather than relying on variable inheritance.
         * ------------------------------------------------------------- */

        /* Top toolbar / header bar + the thin decoration strip */
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            background: var(--tm-paper) !important;
        }
        header[data-testid="stHeader"] * ,
        [data-testid="stToolbar"] * {
            color: var(--tm-navy) !important;
            fill: var(--tm-navy) !important;
        }

        /* Native widgets: force legible colors regardless of the
           deployed theme, since components below have their own
           explicit backgrounds and shouldn't inherit page text color. */
        section[data-testid="stSidebar"] {
            background: var(--tm-paper-2) !important;
            color: var(--tm-ink) !important;
            min-width: 260px !important;
        }
        section[data-testid="stSidebar"] * { color: var(--tm-ink) !important; }
        /* Prevent labels from being truncated/clipped when the sidebar
           is narrow — wrap onto a second line instead of overflowing. */
        section[data-testid="stSidebar"] * {
            white-space: normal !important;
            overflow-wrap: break-word;
            word-break: break-word;
        }

        div[data-testid="stExpander"] {
            background: var(--tm-paper);
        }
        div[data-testid="stExpander"] summary {
            background: var(--tm-paper) !important;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary * {
            color: var(--tm-navy) !important;
        }
        div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"],
        div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] * {
            color: var(--tm-ink) !important;
        }

        .stTextArea textarea {
            background: #FFFFFF;
            color: var(--tm-ink);
            border: 1px solid var(--tm-line);
        }

        .stApp code, .stApp pre {
            background: var(--tm-paper-2) !important;
            color: var(--tm-navy) !important;
        }

        /* st.text() renders via a dedicated testid, not code/pre — theme
           it explicitly so it doesn't inherit dark-theme light text. */
        div[data-testid="stText"], div[data-testid="stText"] * {
            color: var(--tm-ink) !important;
        }

        /* Alerts: st.success / st.info / st.warning / st.error */
        div[data-testid="stAlert"],
        div[data-testid="stAlert"] * {
            color: var(--tm-ink) !important;
        }

        /* Widget labels outside the sidebar, e.g. the trip-description
           textarea label ("Describe your trip"), or slider labels. */
        div[data-testid="stWidgetLabel"],
        div[data-testid="stWidgetLabel"] * {
            color: var(--tm-ink) !important;
        }

        /* st.status() header + body text */
        div[data-testid="stStatusWidget"],
        div[data-testid="stStatusWidget"] * {
            color: var(--tm-ink) !important;
        }

        /* Plain markdown containers (final answer, itinerary body, budget
           write-ups). Skip any container that holds one of our custom
           cards so their own navy/amber/teal colors aren't flattened. */
        div[data-testid="stMarkdownContainer"]:not(:has(.tm-hero)):not(:has(.fc-card)):not(:has(.hc-card)):not(:has(.tm-metric-row)):not(:has(.tm-leg-header)):not(:has(.tm-budget-total)):not(:has(.tm-flow)) {
            color: var(--tm-ink) !important;
        }
        div[data-testid="stMarkdownContainer"]:not(:has(.tm-hero)):not(:has(.fc-card)):not(:has(.hc-card)):not(:has(.tm-metric-row)):not(:has(.tm-leg-header)):not(:has(.tm-budget-total)):not(:has(.tm-flow)) * {
            color: var(--tm-ink) !important;
        }

        /* Buttons: Streamlit renders label text in a nested <p>/<span> that
           carries its own theme color, so descendants need the override too. */
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            border-radius: 8px !important;
            border: 1px solid var(--tm-navy) !important;
            background: #FFFFFF !important;
            color: var(--tm-navy) !important;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(11,37,69,0.12);
        }
        .stButton > button *, .stDownloadButton > button *, .stFormSubmitButton > button * {
            color: inherit !important;
        }
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
            background: var(--tm-navy) !important;
            border-color: var(--tm-navy) !important;
            color: #F5F1E6 !important;
        }
        [data-testid="stLinkButton"] a {
            border-radius: 8px;
            border: 1px solid var(--tm-navy);
            color: var(--tm-navy) !important;
        }
        [data-testid="stLinkButton"] a * { color: inherit !important; }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1120px;
        }

        /* ---------------- Hero: departure board ---------------- */
        .tm-hero {
            padding: 2.1rem 2.3rem;
            border-radius: 16px;
            background:
                radial-gradient(circle at 85% 15%, rgba(232,163,61,0.16), transparent 45%),
                linear-gradient(135deg, var(--tm-navy) 0%, var(--tm-navy-2) 70%);
            color: #F5F1E6;
            margin-bottom: 1.6rem;
            border: 1px solid rgba(232,163,61,0.35);
        }
        .tm-hero-eyebrow {
            font-family: var(--tm-font-mono);
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            color: var(--tm-amber);
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.6rem;
        }
        .tm-hero-eyebrow .tm-dot {
            width: 6px; height: 6px; border-radius: 50%;
            background: var(--tm-amber);
            display: inline-block;
        }
        .tm-hero h1 {
            margin: 0;
            font-family: var(--tm-font-display);
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.01em;
        }
        .tm-hero p {
            margin: 0.5rem 0 0 0;
            color: #C9D3E4;
            font-size: 1rem;
            max-width: 640px;
        }
        .tm-badge {
            display: inline-block;
            padding: 0.18rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(245,241,230,0.25);
            background: rgba(245,241,230,0.06);
            font-family: var(--tm-font-mono);
            font-size: 0.7rem;
            letter-spacing: 0.04em;
            margin-right: 0.4rem;
        }

        /* ---------------- Tabs ---------------- */
        .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--tm-line); }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.55rem 1.1rem;
            font-family: var(--tm-font-display);
            font-weight: 600;
        }
        /* The tab label text sits in a nested <p>, which carries its own
           theme color with higher specificity than the rule above — so
           target every descendant explicitly, not just the tab button. */
        .stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] * {
            color: var(--tm-slate) !important;
        }
        .stTabs [aria-selected="true"], .stTabs [aria-selected="true"] * {
            color: var(--tm-navy) !important;
        }

        /* ---------------- Metric strip ---------------- */
        .tm-metric-row { display: flex; gap: 0.8rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
        .tm-metric-card {
            flex: 1 1 160px;
            background: var(--tm-paper);
            border: 1px solid var(--tm-line);
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }
        .tm-metric-label {
            font-family: var(--tm-font-mono);
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--tm-slate);
        }
        .tm-metric-value {
            font-family: var(--tm-font-mono);
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--tm-navy);
            margin-top: 0.15rem;
        }

        /* ---------------- Leg header (gate signage) ---------------- */
        .tm-leg-header {
            display: flex;
            align-items: baseline;
            gap: 0.6rem;
            margin: 1.4rem 0 0.7rem 0;
            padding-bottom: 0.35rem;
            border-bottom: 2px solid var(--tm-amber);
        }
        .tm-leg-header .tm-leg-icon { font-size: 1rem; }
        .tm-leg-header .tm-leg-text {
            font-family: var(--tm-font-mono);
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.02em;
            color: var(--tm-navy);
            text-transform: uppercase;
        }

        /* ---------------- Card entrance animation ---------------- */
        @keyframes tmFadeInUp {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* ---------------- Boarding-pass flight card ---------------- */
        .fc-card {
            position: relative;
            display: flex;
            background: var(--tm-paper);
            border: 1px solid var(--tm-line);
            border-radius: 12px;
            margin-bottom: 0.9rem;
            overflow: hidden;
            animation: tmFadeInUp 0.35s ease both;
            transition: box-shadow 0.15s ease, transform 0.15s ease;
        }
        .fc-card:hover {
            box-shadow: 0 8px 20px rgba(11,37,69,0.10);
            transform: translateY(-2px);
        }
        .fc-cheapest { border-color: var(--tm-teal); box-shadow: 0 0 0 1px var(--tm-teal) inset; }
        .fc-main { flex: 1; padding: 1rem 1.1rem; min-width: 0; }
        .fc-top { display: flex; justify-content: space-between; align-items: baseline; gap: 0.5rem; flex-wrap: wrap; }
        .fc-airline {
            font-family: var(--tm-font-display);
            font-weight: 700;
            font-size: 1.02rem;
            color: var(--tm-navy);
        }
        .fc-route {
            font-family: var(--tm-font-mono);
            color: var(--tm-slate);
            font-size: 0.82rem;
            margin-top: 0.5rem;
            line-height: 1.7;
        }
        .fc-stub {
            position: relative;
            width: 148px;
            flex-shrink: 0;
            background: var(--tm-navy);
            color: #F5F1E6;
            padding: 1rem 0.9rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: flex-start;
        }
        .fc-stub::before {
            /* perforation line */
            content: "";
            position: absolute;
            top: 0; bottom: 0; left: 0;
            border-left: 2px dashed rgba(245,241,230,0.35);
        }
        .fc-stub::after {
            content: "";
            position: absolute;
            top: -9px; left: -9px;
            width: 18px; height: 18px;
            border-radius: 50%;
            background: var(--tm-paper);
            box-shadow: 0 148px 0 0 var(--tm-paper);
        }
        .fc-price-label {
            font-family: var(--tm-font-mono);
            font-size: 0.65rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--tm-amber);
        }
        .fc-price {
            font-family: var(--tm-font-mono);
            font-weight: 700;
            font-size: 1.05rem;
            margin-top: 0.15rem;
            line-height: 1.3;
        }
        .fc-badge {
            display: inline-block;
            font-family: var(--tm-font-mono);
            font-size: 0.65rem;
            letter-spacing: 0.04em;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            background: var(--tm-teal);
            color: #F5F1E6;
            margin-top: 0.55rem;
        }

        /* Small "data source" pill shown on flight/hotel cards */
        .tm-src-badge {
            display: inline-block;
            font-family: var(--tm-font-mono);
            font-size: 0.6rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 0.12rem 0.45rem;
            border-radius: 999px;
            border: 1px solid var(--tm-line);
            color: var(--tm-slate);
            background: var(--tm-paper-2);
        }

        /* Mobile: stack the boarding-pass card instead of clipping the stub */
        @media (max-width: 640px) {
            .fc-card { flex-direction: column; }
            .fc-stub { width: 100%; align-items: flex-start; }
            .fc-stub::before { display: none; }
            .fc-stub::after { display: none; }
        }

        /* ---------------- Luggage-tag hotel card ---------------- */
        .hc-card {
            position: relative;
            border: 1px dashed var(--tm-line);
            border-radius: 12px;
            padding: 1rem 1.1rem 1rem 1.4rem;
            margin-bottom: 0.9rem;
            background: var(--tm-paper);
            animation: tmFadeInUp 0.35s ease both;
            transition: box-shadow 0.15s ease, transform 0.15s ease;
        }
        .hc-card:hover {
            box-shadow: 0 8px 20px rgba(11,37,69,0.10);
            transform: translateY(-2px);
        }
        .hc-card::before {
            /* tag hole */
            content: "";
            position: absolute;
            top: 1rem; left: -7px;
            width: 14px; height: 14px;
            border-radius: 50%;
            background: #FFFFFF;
            border: 1px solid var(--tm-line);
        }
        .hc-city {
            font-family: var(--tm-font-mono);
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--tm-amber);
            background: var(--tm-navy);
            display: inline-block;
            padding: 0.1rem 0.5rem;
            border-radius: 5px;
            margin-bottom: 0.4rem;
        }
        .hc-title { font-weight: 700; color: var(--tm-navy); font-size: 1rem; margin-bottom: 0.2rem; }
        .hc-snippet { color: var(--tm-slate); font-size: 0.87rem; line-height: 1.45; margin-top: 0.2rem; }
        .hc-price {
            font-family: var(--tm-font-mono);
            font-weight: 700;
            color: var(--tm-teal);
            font-size: 0.88rem;
            margin: 0.4rem 0 0.15rem 0;
        }
        .hc-noprice {
            color: var(--tm-slate);
            font-size: 0.8rem;
            font-style: italic;
            margin: 0.4rem 0 0.15rem 0;
        }
        .hc-link {
            display: inline-block;
            margin-top: 0.5rem;
            font-family: var(--tm-font-mono);
            font-size: 0.76rem;
            font-weight: 600;
            color: var(--tm-navy);
            text-decoration: none;
            border-bottom: 1px solid var(--tm-amber);
        }
        .hc-link:hover { color: var(--tm-amber); }

        /* ---------------- Itinerary / expanders ---------------- */
        div[data-testid="stExpander"] {
            border: 1px solid var(--tm-line);
            border-radius: 10px;
            background: var(--tm-paper);
        }
        div[data-testid="stExpander"] summary {
            font-family: var(--tm-font-display);
            font-weight: 700;
            color: var(--tm-navy);
        }

        /* ---------------- Budget ---------------- */
        .tm-budget-total {
            font-family: var(--tm-font-mono);
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--tm-navy);
        }

        /* ---------------- Architecture / flow diagram ---------------- */
        .tm-flow {
            display: flex;
            align-items: stretch;
            gap: 0.4rem;
            flex-wrap: wrap;
            margin: 0.6rem 0 1rem 0;
        }
        .tm-flow-step {
            flex: 1 1 150px;
            background: var(--tm-paper);
            border: 1px solid var(--tm-line);
            border-radius: 10px;
            padding: 0.7rem 0.8rem;
            position: relative;
        }
        .tm-flow-step .tm-flow-icon { font-size: 1.15rem; }
        .tm-flow-step .tm-flow-title {
            font-family: var(--tm-font-display);
            font-weight: 700;
            color: var(--tm-navy);
            font-size: 0.88rem;
            margin-top: 0.25rem;
        }
        .tm-flow-step .tm-flow-desc {
            font-family: var(--tm-font-mono);
            color: var(--tm-slate);
            font-size: 0.68rem;
            margin-top: 0.2rem;
            line-height: 1.4;
        }
        .tm-flow-arrow {
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--tm-amber);
            font-size: 1.1rem;
            flex: 0 0 20px;
        }
        @media (max-width: 760px) {
            .tm-flow-arrow { display: none; }
        }

        /* ---------------- Alerts ---------------- */
        div[data-testid="stAlert"] {
            border-radius: 10px;
            border: 1px solid var(--tm-line);
        }

        .tm-footer {
            text-align: center;
            color: var(--tm-slate);
            font-family: var(--tm-font-mono);
            font-size: 0.75rem;
            letter-spacing: 0.03em;
            margin-top: 2.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Session state
# =========================================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"user_{uuid.uuid4().hex[:12]}"
if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# Helpers — work directly on structured data, no parsing
# (unchanged from the structured-data version)
# =========================================================
def group_offers_by_leg(offers: list):
    """Groups flight offers by their leg_label (e.g. 'BOM -> JFK on 2026-08-20')."""
    groups = defaultdict(list)
    for o in offers:
        groups[o.get("leg_label", "Unknown leg")].append(o)
    for leg in groups:
        groups[leg].sort(key=lambda o: o.get("price_amount") or float("inf"))
    return dict(groups)


def cheapest_per_leg(offers: list):
    """Returns {leg_label: cheapest_offer_dict} — used for budget totals."""
    groups = group_offers_by_leg(offers)
    return {leg: offers_list[0] for leg, offers_list in groups.items() if offers_list}


def compute_budget(flight_offers: list, hotel_results: list, nights: int = 6):
    """
    Builds a budget breakdown straight from structured data —
    cheapest fare per leg + average of any hotels with a real price.
    Returns dict of {category: amount_inr}, or {} if nothing to compute.
    """
    budget = {}

    cheapest = cheapest_per_leg(flight_offers)
    flight_total = sum(
        o.get("price_converted") or 0
        for o in cheapest.values()
        if o.get("price_converted")
    )
    if flight_total:
        budget["Flights"] = round(flight_total, 2)

    priced_hotels = [h for h in hotel_results if h.get("price_inr")]
    if priced_hotels:
        avg_nightly = sum(h["price_inr"]["amount_inr"] for h in priced_hotels) / len(priced_hotels)
        budget["Accommodation"] = round(avg_nightly * nights, 2)

    return budget


# =========================================================
# Live progress wrapper around the (currently blocking) agent call
# ---------------------------------------------------------
# backend.run_travel_agent() doesn't yet expose a streaming/callback API,
# so we run it on a background thread and cycle through the known pipeline
# stages in st.status() while we wait. This is an honest "still working,
# here's roughly what's happening" indicator rather than a claim of exact
# real-time agent state. Swap the `while thread.is_alive()` loop for a
# real event stream the moment backend supports one.
# =========================================================
PIPELINE_STAGES = [
    ("🛫", "Flight Agent checking bookable fares (Duffel)..."),
    ("📡", "Cross-checking live flight status (AviationStack)..."),
    ("🏨", "Hotel Agent scanning listings (Tavily)..."),
    ("🗓️", "Itinerary Agent drafting your day-by-day plan..."),
    ("✨", "Final Agent polishing the summary..."),
]


def run_with_progress(user_input: str, thread_id: str):
    outcome = {}

    def _worker():
        try:
            outcome["result"] = run_travel_agent(user_input=user_input, thread_id=thread_id)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI below
            outcome["error"] = exc

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    with st.status("Planning your trip...", expanded=True) as status:
        stage_idx = 0
        while worker.is_alive():
            icon, label = PIPELINE_STAGES[stage_idx % len(PIPELINE_STAGES)]
            status.update(label=f"{icon} {label}")
            time.sleep(1.5)
            stage_idx += 1
        worker.join()

        if "error" in outcome:
            status.update(label="⚠️ Planning failed", state="error")
            raise outcome["error"]

        status.update(label="✅ Trip plan ready", state="complete")

    return outcome.get("result")


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.markdown("## ✈️ TripMate AI")
    st.caption("Multi-agent travel planner powered by LangGraph")
    st.divider()
    st.markdown("**Session**")
    st.code(st.session_state.thread_id, language=None)
    if st.button("🔄 New session", use_container_width=True):
        st.session_state.thread_id = f"user_{uuid.uuid4().hex[:12]}"
        st.session_state.result = None
        st.rerun()
    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        """
        1. **Flight Agent** — bookable fares (Duffel) per leg + live status (AviationStack)
        2. **Hotel Agent** — hotel options per city (Tavily search)
        3. **Itinerary Agent** — day-by-day plan (LLM)
        4. **Final Agent** — polished response (LLM)
        """
    )
    st.divider()
    if st.session_state.history:
        st.markdown("**Recent requests**")
        for q in reversed(st.session_state.history[-5:]):
            st.caption(f"• {q}")

# =========================================================
# Hero
# =========================================================
render_html(
    """
    <div class="tm-hero">
        <div class="tm-hero-eyebrow"><span class="tm-dot"></span>NOW BOARDING · TRIP PLANNER</div>
        <h1>Plan your next trip in seconds</h1>
        <p>Tell TripMate where you're going, and let the agents handle flights, hotels, and the day-by-day plan.</p>
        <div style="margin-top:0.9rem;">
            <span class="tm-badge">✈ FLIGHTS</span>
            <span class="tm-badge">🏨 HOTELS</span>
            <span class="tm-badge">🗓 ITINERARY</span>
        </div>
    </div>
    """
)

# =========================================================
# Input form
# =========================================================
with st.form("trip_form", clear_on_submit=False):
    user_input = st.text_area(
        "Describe your trip",
        placeholder="e.g. Plan a 7 day trip from Mumbai to Japan under 2 lakhs, including flights, hotels and sightseeing",
        height=90,
    )
    col1, _ = st.columns([1, 5])
    with col1:
        submitted = st.form_submit_button("🧭 Plan my trip", use_container_width=True, type="primary")

if submitted:
    if not user_input or not user_input.strip():
        st.warning("Please describe your trip before submitting.")
    else:
        try:
            result = run_with_progress(user_input.strip(), st.session_state.thread_id)
            st.session_state.result = result
            st.session_state.history.append(user_input.strip())
        except Exception as e:
            st.session_state.result = None
            st.error("Something went wrong while planning your trip.")
            with st.expander("Error details"):
                st.code(str(e))
            st.markdown(
                "Try: shortening the request, double-checking city/airport names, "
                "or loosening the budget — then submit again."
            )

# =========================================================
# Results
# =========================================================
result = st.session_state.result

if result:
    st.success("Your trip plan is ready.")

    render_html(
        f"""
        <div class="tm-metric-row">
            <div class="tm-metric-card">
                <div class="tm-metric-label">🧠 LLM calls used</div>
                <div class="tm-metric-value">{result.get("llm_calls", 0)}</div>
            </div>
            <div class="tm-metric-card">
                <div class="tm-metric-label">🔑 Session</div>
                <div class="tm-metric-value" style="font-size:0.95rem;">{result.get("thread_id", "-")}</div>
            </div>
        </div>
        """
    )

    with st.expander("🧩 How this was built"):
        render_html(
            """
            <div class="tm-flow">
                <div class="tm-flow-step">
                    <div class="tm-flow-icon">🛫</div>
                    <div class="tm-flow-title">Flight Agent</div>
                    <div class="tm-flow-desc">Duffel fares<br>+ AviationStack status</div>
                </div>
                <div class="tm-flow-arrow">→</div>
                <div class="tm-flow-step">
                    <div class="tm-flow-icon">🏨</div>
                    <div class="tm-flow-title">Hotel Agent</div>
                    <div class="tm-flow-desc">Tavily search<br>per city</div>
                </div>
                <div class="tm-flow-arrow">→</div>
                <div class="tm-flow-step">
                    <div class="tm-flow-icon">🗓️</div>
                    <div class="tm-flow-title">Itinerary Agent</div>
                    <div class="tm-flow-desc">LLM day-by-day<br>planning</div>
                </div>
                <div class="tm-flow-arrow">→</div>
                <div class="tm-flow-step">
                    <div class="tm-flow-icon">✨</div>
                    <div class="tm-flow-title">Final Agent</div>
                    <div class="tm-flow-desc">LLM polish<br>+ summary</div>
                </div>
            </div>
            """
        )
        st.caption(
            "Orchestrated as a LangGraph state machine — each node reads/writes shared "
            "state, so later agents (itinerary, final) see the structured flight and "
            "hotel data straight from the earlier nodes rather than re-parsing text."
        )

    flight_offers = result.get("flight_offers", [])
    hotel_results = result.get("hotel_results", [])
    live_status = result.get("live_status", "")

    tab_final, tab_flights, tab_hotels, tab_itinerary, tab_budget = st.tabs(
        ["📋 Summary", "🛫 Flights", "🏨 Hotels", "🗓️ Itinerary", "💰 Budget"]
    )

    # ---------- Final Summary ----------
    with tab_final:
        st.markdown(result.get("answer", "No response generated."))
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "⬇️ Download plan as Markdown",
                data=result.get("answer", ""),
                file_name="tripmate_itinerary.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                "⬇️ Download full data as JSON",
                data=json.dumps(result, indent=2, default=str),
                file_name="tripmate_result.json",
                mime="application/json",
                use_container_width=True,
                help="Every flight offer, hotel result, and the raw agent output — useful for re-loading or sharing this plan.",
            )

    # ---------- Flights ----------
    with tab_flights:
        if flight_offers:
            grouped = group_offers_by_leg(flight_offers)
            for leg_label, offers in grouped.items():
                render_html(
                    f"""
                    <div class="tm-leg-header">
                        <span class="tm-leg-icon">🛫</span>
                        <span class="tm-leg-text">{leg_label}</span>
                    </div>
                    """
                )
                for i, o in enumerate(offers):
                    price_display = f"{o['price_amount']:.2f} {o['price_currency']}" if o.get("price_amount") else "Price unavailable"
                    converted_line = f"~₹{o['price_converted']:,.2f}" if o.get("price_converted") else ""

                    route_lines = "<br>".join(
                        f"{s['dep']} → {s['arr']} · Flight {s['flight_no']} · Dep {s['dep_time']} · Arr {s['arr_time']}"
                        for s in o.get("segments", [])
                    ) or "Route details unavailable"

                    card_class = "fc-card fc-cheapest" if i == 0 else "fc-card"
                    cheapest_badge = '<span class="fc-badge">CHEAPEST</span>' if i == 0 else ""

                    render_html(
                        f"""
                        <div class="{card_class}">
                            <div class="fc-main">
                                <div class="fc-top">
                                    <span class="fc-airline">{o['airline']}</span>
                                    <span class="tm-src-badge">Duffel</span>
                                </div>
                                <div class="fc-route">{route_lines}</div>
                                {cheapest_badge}
                            </div>
                            <div class="fc-stub">
                                <div class="fc-price-label">Fare</div>
                                <div class="fc-price">{price_display}</div>
                                <div class="fc-price-label" style="margin-top:0.3rem; color:#C9D3E4;">{converted_line}</div>
                            </div>
                        </div>
                        """
                    )
        else:
            st.info("No bookable fares found for this trip.")
            st.markdown(
                """
                A few things that usually help:
                - Try nearby airports instead of the exact city named
                - Shift travel dates by a few days — fare availability is sparse for some routes
                - Double-check the origin/destination were understood correctly in your request
                """
            )

        if live_status:
            with st.expander("📡 Live flight status (AviationStack)"):
                st.text(live_status)

    # ---------- Hotels ----------
    with tab_hotels:
        if hotel_results:
            # Priced hotels first
            hotels_sorted = sorted(hotel_results, key=lambda h: h.get("price_inr") is None)
            cols = st.columns(2)
            for i, h in enumerate(hotels_sorted):
                with cols[i % 2]:
                    link_html = (
                        f'<a class="hc-link" href="{h["url"]}" target="_blank">View listing →</a>'
                        if h.get("url") else ""
                    )
                    if h.get("price_inr"):
                        price_html = f'<div class="hc-price">~₹{h["price_inr"]["amount_inr"]:,.0f}/night (from {h["price_inr"]["source_currency"]})</div>'
                    else:
                        price_html = '<div class="hc-noprice">Price not listed</div>'

                    city_html = f'<span class="hc-city">{h["city"]}</span>' if h.get("city") else ""

                    render_html(
                        f"""
                        <div class="hc-card">
                            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:0.4rem;">
                                {city_html}
                                <span class="tm-src-badge">Tavily</span>
                            </div>
                            <div class="hc-title">{h['title']}</div>
                            {price_html}
                            <div class="hc-snippet">{h['snippet']}</div>
                            {link_html}
                        </div>
                        """
                    )
        else:
            st.info("No hotel results found for this trip.")
            st.markdown(
                """
                A few things that usually help:
                - Confirm the destination city was named clearly in your request
                - Try again — hotel search results can vary run to run
                """
            )

    # ---------- Itinerary ----------
    with tab_itinerary:
        itinerary_text = result.get("itinerary", "")
        if itinerary_text:
            days = itinerary_text.split("### Day") if "### Day" in itinerary_text else None
            if days and len(days) > 1:
                intro = days[0]
                if intro.strip():
                    st.markdown(intro)

                day_labels, day_bodies = [], []
                for d in days[1:]:
                    header_line = d.split("\n", 1)[0].strip()
                    body = d.split("\n", 1)[1] if "\n" in d else ""
                    short_label = header_line.split(":", 1)[0].strip()
                    day_labels.append(f"Day {short_label}" if short_label else "Day")
                    day_bodies.append(body)

                day_tabs = st.tabs(day_labels)
                for day_tab, body in zip(day_tabs, day_bodies):
                    with day_tab:
                        st.markdown(body)
            else:
                st.markdown(itinerary_text)
        else:
            st.info("No itinerary available.")

    # ---------- Budget ----------
    with tab_budget:
        # The backend's final agent already estimates a full breakdown
        # (Flights, Accommodation, Food & Dining, Sightseeing & Activities,
        # Local Transportation, Miscellaneous, Travel Insurance, etc.) — use
        # that as the source of truth rather than recomputing from scratch,
        # since compute_budget() below only knows how to derive Flights and
        # Accommodation from raw offer data and would otherwise drop every
        # other category the backend estimated.
        base_budget = result.get("budget_breakdown", {})
        if not base_budget and (flight_offers or hotel_results):
            base_budget = compute_budget(flight_offers, hotel_results, nights=6)

        if base_budget:
            slider_col1, slider_col2 = st.columns(2)
            with slider_col1:
                nights = st.slider("Nights", min_value=1, max_value=21, value=6)
            with slider_col2:
                travelers = st.slider("Travelers", min_value=1, max_value=8, value=1)

            budget = dict(base_budget)

            # Flights and Accommodation can be live-recomputed against the
            # nights slider since we have real priced data for them. Every
            # other category (food, activities, local transport, insurance,
            # etc.) stays at the backend's original estimate — we don't have
            # a per-night basis to rescale those ourselves.
            live_recompute = compute_budget(flight_offers, hotel_results, nights=nights)
            for category in ("Flights", "Accommodation"):
                if category in live_recompute:
                    budget[category] = live_recompute[category]

            # Scale every category by traveler count.
            budget = {category: amount * travelers for category, amount in budget.items()}

            if budget:
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=list(budget.keys()),
                            values=list(budget.values()),
                            hole=0.55,
                            marker=dict(
                                colors=["#0B2545", "#E8A33D", "#1F8A70", "#D2564F", "#647188"],
                                line=dict(color="#FBF8F2", width=2),
                            ),
                            textfont=dict(family="IBM Plex Mono", size=13),
                        )
                    ]
                )
                fig.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=True,
                    height=380,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Manrope", color="#1C2530", size=13),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        font=dict(family="Manrope", color="#1C2530", size=12),
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

                total = sum(budget.values())
                render_html(
                    f'<div class="tm-metric-label">Estimated total (INR) · {nights} nights · {travelers} traveler(s)</div>'
                    f'<div class="tm-budget-total">₹{total:,.0f}</div>'
                )

                with st.expander("See breakdown"):
                    for category, amount in budget.items():
                        st.write(f"**{category}:** ₹{amount:,.2f}")

                st.caption(
                    "Flights and Accommodation recompute live from priced offers as you move "
                    "the Nights slider (cheapest bookable fare per leg, and the average nightly "
                    "hotel rate). Other categories are the agent's original per-trip estimate. "
                    "All categories scale with the Travelers slider."
                )
            else:
                st.info("Not enough data to estimate a budget for this trip.")
        else:
            st.info("Budget breakdown unavailable for this trip.")

else:
    st.info("Enter a trip request above to get started — e.g. *\"7 day trip from Delhi to Bali under ₹1.5 lakh\"*.")

# =========================================================
# Footer
# =========================================================
render_html(
    '<div class="tm-footer">TRIPMATE AI · BUILT WITH LANGGRAPH · GROQ · DUFFEL · AVIATIONSTACK · TAVILY</div>'
)