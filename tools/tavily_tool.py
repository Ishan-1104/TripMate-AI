from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def tavily_Search(query, max_results=5, search_depth="advanced", include_domains=None):
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,          # "advanced" digs deeper than default "basic"
        include_domains=include_domains,     # bias toward real listings, not blogs
    )

    results = []

    for i, r in enumerate(response["results"], 1):
        title   = r.get("title", "Unknown")
        url     = r.get("url", "")
        snippet = r.get("content", "").strip()
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        results.append(f"{i}. **{title}**\n   {url}\n   {snippet}")

    return "\n\n".join(results)

import re
from tools.duffel_client import convert_amount  # reuse existing frankfurter-based conversion

# Order matters: multi-char symbols MUST come before their single-char substrings
CURRENCY_SYMBOL_MAP = {
    "R$": "BRL", "US$": "USD", "C$": "CAD", "A$": "AUD",
    "₹": "INR", "₱": "PHP", "€": "EUR", "£": "GBP", "$": "USD",
}
PRICE_PATTERN = re.compile(
    r"(R\$|US\$|C\$|A\$|₹|₱|€|£|\$|USD|INR|EUR|GBP|CAD|AUD)\s?([\d,]+(?:\.\d+)?)"
)

def extract_price_inr(snippet: str):
    """Finds a price in a snippet and converts it to INR, tagging the real source currency."""
    m = PRICE_PATTERN.search(snippet)
    if not m:
        return None
    symbol, amount_str = m.groups()
    currency = CURRENCY_SYMBOL_MAP.get(symbol, symbol)
    try:
        amount = float(amount_str.replace(",", ""))
    except ValueError:
        return None

    if currency == "INR":
        return {"amount_inr": round(amount, 2), "source_currency": "INR"}

    converted = convert_amount(amount, currency, "INR")
    if converted is None:
        return None
    return {"amount_inr": converted, "source_currency": currency}


def tavily_search_structured(query, max_results=5, search_depth="advanced", include_domains=None):
    response = client.search(
        query=query, max_results=max_results,
        search_depth=search_depth, include_domains=include_domains,
    )
    results = []
    for r in response.get("results", []):
        snippet = r.get("content", "").strip()
        price_info = extract_price_inr(snippet)  # dict with amount_inr + source_currency, or None

        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        results.append({
            "title": r.get("title", "Unknown"),
            "url": r.get("url", ""),
            "snippet": snippet,
            "price_inr": price_info,  # {"amount_inr": ..., "source_currency": "BRL"} or None
        })
    return results