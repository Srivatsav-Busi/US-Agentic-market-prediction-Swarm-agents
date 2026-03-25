import json
import logging
import re
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

GSTACK_BIN = Path.home() / ".gemini/skills/gstack/browse/dist/browse"

SECTORS = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB"
}

def fetch_sector_data() -> dict:
    """
    Uses the gstack headless browser to scrape real-time pricing for 11 S&P 500 sectors.
    """
    results = {}
    if not GSTACK_BIN.exists():
        logger.error(f"gstack binary not found at {GSTACK_BIN}")
        return results

    for sector, ticker in SECTORS.items():
        try:
            # Navigate to the Yahoo Finance page for the ETF
            subprocess.run(
                [str(GSTACK_BIN), "goto", f"https://finance.yahoo.com/quote/{ticker}"],
                check=True,
                capture_output=True
            )
            
            # Wait briefly to ensure content is loaded
            time.sleep(1)
            
            # Get the page text instead of relying on specific DOM elements
            res_text = subprocess.run(
                [str(GSTACK_BIN), "text"],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Use regex to find the ticker, followed by its full name, price, and percentage change.
            # E.g., "XLK State Street Technology Select Sector SPDR ETF  137.56  +0.56%"
            pattern = re.compile(rf"{ticker}\b.*?([\d\.,]+)\s+([+-][\d\.,]+)%", re.IGNORECASE)
            match = pattern.search(res_text.stdout)
            
            if match:
                price_str = match.group(1).replace(',', '')
                change_str = match.group(2).replace(',', '')
                results[sector] = {
                    "ticker": ticker,
                    "price": float(price_str),
                    "change_pct": float(change_str)
                }
                logger.info(f"Successfully scraped {sector} ({ticker}): {price_str}")
            else:
                logger.warning(f"Regex failed to match price in text output for {sector} ({ticker})")
                results[sector] = {"ticker": ticker, "error": "Regex match failed"}
                
        except Exception as e:
            logger.warning(f"Failed to scrape data for {sector} ({ticker}): {e}")
            results[sector] = {"ticker": ticker, "error": str(e)}
            
    return results

if __name__ == "__main__":
    print(json.dumps(fetch_sector_data(), indent=2))
