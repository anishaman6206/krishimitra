# app/services/mandi.py
import sys
import os
from typing import Optional, Dict, Any, List

from backend.app.tools.mandi_cached import latest_price_cached as mandi_price
from backend.app.tools.mandi_cached import price_series_cached as mandi_price_series

class MandiService:
    """
    Wrapper around mandi/market tools.

    Primary inputs for MVP:
      - commodity (required)
      - state, district (recommended)

    Optional refinements for advanced users:
      - market (specific mandi)
      - variety, grade (quality specifics)
      - strict_state (default True; relax to nationwide when False)
    """

    def __init__(self) -> None:
        pass

    async def get_market_info(
        self,
        commodity: str,
        state: Optional[str] = None,
        district: Optional[str] = None,
        market: Optional[str] = None,
        variety: Optional[str] = None,
        grade: Optional[str] = None,
        strict_state: bool = True,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Get latest market information for a commodity.

        Returns a normalized dict from the cached tool, or {"error": "..."} on failure.
        """
        try:
            result = await mandi_price(
                commodity,
                state=state,
                district=district,
                market=market,
                variety=variety,
                grade=grade,
                strict_state=strict_state,
                debug=debug,
            )
            return result or {}
        except Exception as e:
            print(f"Mandi info error: {e}")
            return {"error": str(e)}

    async def get_price_series(
        self,
        commodity: str,
        state: Optional[str] = None,
        district: Optional[str] = None,
        market: Optional[str] = None,
        days: int = 90,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return [{'date': 'YYYY-MM-DD', 'modal_price': int}, ...] for plotting trends.
        """
        try:
            series = await mandi_price_series(
                commodity=commodity,
                state=state,
                district=district,
                market=market,
                days=days,
                debug=debug,
            )
            # Always return a list (even if error dict bubbles up)
            return series if isinstance(series, list) else []
        except Exception as e:
            print(f"Mandi series error: {e}")
            return []
