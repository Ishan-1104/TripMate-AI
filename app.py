"""
TripMate AI — Streamlit UI
Frontend for the LangGraph-based multi-agent travel planner in backend.py
(flight_agent -> hotel_agent -> itinerary_agent -> final_agent)

This version consumes STRUCTURED data returned by run_travel_agent()
(flight_offers: list[dict], hotel_results: list[dict]) directly —
no regex parsing of formatted text.
"""

import uuid
from collections import defaultdict

import streamlit as st
import plotly.graph_objects as go

from backend import run_travel_agent

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
# Styling
# =========================================================
st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1100px;
        }
        .tm-hero {
            padding: 1.75rem 2rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #1e3a5f 0%, #2c5f8a 60%, #3d7ea6 100%);
            color: #ffffff;
            margin-bottom: 1.5rem;
        }
        .tm-hero h1 { margin: 0; font-size: 2rem; font-weight: 700; }
        .tm-hero p { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 1rem; }
        .tm-badge {
            display: inline-block;
            padding: 0.15rem 0.65rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.15);
            font-size: 0.75rem;
            margin-right: 0.4rem;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 0.5rem 1rem; }
        div[data-testid="stMetricValue"] { font-size: 1.4rem; }

        .fc-card {
            border: 1px solid #e6e6e6;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            background: #ffffff;
        }
        .fc-top { display: flex; justify-content: space-between; align-items: center; }
        .fc-airline { font-weight: 700; font-size: 1.05rem; color: #1e3a5f; }
        .fc-price { font-weight: 700; font-size: 1.15rem; color: #17795e; }
        .fc-route { color: #555; font-size: 0.9rem; margin-top: 0.35rem; }
        .fc-leg-label { color: #888; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.3rem; }
        .fc-badge {
            display: inline-block;
            font-size: 0.7rem;
            padding: 0.1rem 0.5rem;
            border-radius: 6px;
            background: #eef4fb;
            color: #2c5f8a;
            margin-top: 0.5rem;
        }
        .fc-cheapest { border-color: #17795e; box-shadow: 0 0 0 1px #17795e inset; }

        .hc-card {
            border: 1px solid #e6e6e6;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            background: #fff;
        }
        .hc-title { font-weight: 700; color: #1e3a5f; font-size: 1rem; margin-bottom: 0.2rem; }
        .hc-snippet { color: #555; font-size: 0.88rem; line-height: 1.4; }
        .hc-link { font-size: 0.8rem; }
        .hc-price { font-weight: 700; color: #17795e; font-size: 0.9rem; margin: 0.3rem 0; }
        .hc-noprice { color: #999; font-size: 0.8rem; font-style: italic; margin: 0.3rem 0; }

        .tm-footer { text-align: center; color: #888; font-size: 0.8rem; margin-top: 2.5rem; }
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
st.markdown(
    """
    <div class="tm-hero">
        <span class="tm-badge">Flights</span>
        <span class="tm-badge">Hotels</span>
        <span class="tm-badge">Itinerary</span>
        <h1>Plan your next trip in seconds</h1>
        <p>Tell TripMate where you're going, and let the agents handle flights, hotels, and the day-by-day plan.</p>
    </div>
    """,
    unsafe_allow_html=True,
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
        with st.spinner("Searching flights & hotels, building your itinerary... this can take up to a minute."):
            try:
                result = run_travel_agent(user_input=user_input.strip(), thread_id=st.session_state.thread_id)
                st.session_state.result = result
                st.session_state.history.append(user_input.strip())
            except Exception as e:
                st.session_state.result = None
                st.error(f"Something went wrong while planning your trip: {e}")

# =========================================================
# Results
# =========================================================
result = st.session_state.result

if result:
    st.success("Your trip plan is ready.")

    m1, m2 = st.columns(2)
    m1.metric("LLM calls used", result.get("llm_calls", 0))
    m2.metric("Session", result.get("thread_id", "-"))

    flight_offers = result.get("flight_offers", [])
    hotel_results = result.get("hotel_results", [])
    live_status = result.get("live_status", "")

    tab_final, tab_flights, tab_hotels, tab_itinerary, tab_budget = st.tabs(
        ["📋 Summary", "🛫 Flights", "🏨 Hotels", "🗓️ Itinerary", "💰 Budget"]
    )

    # ---------- Final Summary ----------
    with tab_final:
        st.markdown(result.get("answer", "No response generated."))
        st.download_button(
            "⬇️ Download plan as Markdown",
            data=result.get("answer", ""),
            file_name="tripmate_itinerary.md",
            mime="text/markdown",
        )

    # ---------- Flights ----------
    with tab_flights:
        if flight_offers:
            grouped = group_offers_by_leg(flight_offers)
            for leg_label, offers in grouped.items():
                st.markdown(f"#### {leg_label}")
                for i, o in enumerate(offers):
                    price_display = f"{o['price_amount']:.2f} {o['price_currency']}" if o.get("price_amount") else "Price unavailable"
                    if o.get("price_converted"):
                        price_display += f" (~₹{o['price_converted']:,.2f})"

                    route_lines = "<br>".join(
                        f"{s['dep']} → {s['arr']} · Flight {s['flight_no']} · Dep {s['dep_time']} · Arr {s['arr_time']}"
                        for s in o.get("segments", [])
                    ) or "Route details unavailable"

                    card_class = "fc-card fc-cheapest" if i == 0 else "fc-card"
                    cheapest_badge = '<span class="fc-badge">CHEAPEST</span>' if i == 0 else ""

                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <div class="fc-top">
                                <span class="fc-airline">{o['airline']}</span>
                                <span class="fc-price">{price_display}</span>
                            </div>
                            <div class="fc-route">{route_lines}</div>
                            {cheapest_badge}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No bookable fares found for this trip. Try a different route or date.")

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

                    city_html = f'<span class="fc-leg-label">{h["city"]}</span><br>' if h.get("city") else ""

                    st.markdown(
                        f"""
                        <div class="hc-card">
                            {city_html}
                            <div class="hc-title">{h['title']}</div>
                            {price_html}
                            <div class="hc-snippet">{h['snippet']}</div>
                            <div style="margin-top:0.4rem;">{link_html}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No hotel results found for this trip.")

    # ---------- Itinerary ----------
    with tab_itinerary:
        itinerary_text = result.get("itinerary", "")
        if itinerary_text:
            days = itinerary_text.split("### Day") if "### Day" in itinerary_text else None
            if days and len(days) > 1:
                if days[0].strip():
                    st.markdown(days[0])
                for d in days[1:]:
                    header_line = d.split("\n", 1)[0].strip()
                    body = d.split("\n", 1)[1] if "\n" in d else ""
                    with st.expander(f"📅 Day{header_line}", expanded=False):
                        st.markdown(body)
            else:
                st.markdown(itinerary_text)
        else:
            st.info("No itinerary available.")

    # ---------- Budget ----------
    # ---------- Budget ----------
    with tab_budget:
        budget = result.get("budget_breakdown", {})
        if budget:
            fig = go.Figure(data=[go.Pie(labels=list(budget.keys()), values=list(budget.values()), hole=0.5)])
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True, height=380)
            st.plotly_chart(fig, use_container_width=True)

            total = sum(budget.values())
            st.metric("Estimated total (INR)", f"₹{total:,.0f}")

            with st.expander("See breakdown"):
                for category, amount in budget.items():
                    st.write(f"**{category}:** ₹{amount:,.2f}")
        else:
            st.info("Budget breakdown unavailable for this trip.")

else:
    st.info("Enter a trip request above to get started — e.g. *\"7 day trip from Delhi to Bali under ₹1.5 lakh\"*.")

# =========================================================
# Footer
# =========================================================
st.markdown(
    '<div class="tm-footer">TripMate AI · Built with LangGraph, Groq, Duffel, AviationStack & Tavily</div>',
    unsafe_allow_html=True,
)