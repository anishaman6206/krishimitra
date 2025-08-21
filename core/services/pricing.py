# core/services/pricing.py
from typing import Optional, Dict, Any

from backend.app.tools.mandi_cached import (
    latest_price_cached,
    advise_sell_or_wait_cached,
)

class PricingService:
    """Wrapper around existing pricing tools."""

    async def get_latest_price(
        self,
        commodity: str,
        district: Optional[str] = None,
        state: Optional[str] = None,
        market: Optional[str] = None,
        variety: Optional[str] = None,
        grade: Optional[str] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        try:
            result = await latest_price_cached(
                commodity,
                district=district,
                state=state,
                market=market,
                variety=variety,
                grade=grade,
                debug=debug,
            )
            return result or {}
        except Exception as e:
            print(f"Pricing error: {e}")
            return {"error": str(e)}

    async def advise_sell_or_wait(
        self,
        commodity: str,
        state: Optional[str] = None,
        district: Optional[str] = None,
        market: Optional[str] = None,
        variety: Optional[str] = None,
        grade: Optional[str] = None,
        horizon_days: Optional[int] = None,
        qty_qtl: Optional[float] = None,
        price_context: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Get sell/wait advice for a commodity.
        Ensures horizon_days is an int (defaults to 7 if None/invalid).
        """
        try:
            H = int(horizon_days) if isinstance(horizon_days, int) and horizon_days > 0 else 7
            result = await advise_sell_or_wait_cached(
                commodity=commodity,
                state=state,
                district=district,
                market=market,
                variety=variety,
                grade=grade,
                horizon_days=H,
                qty_qtl=qty_qtl,
                price_context=price_context,
                debug=debug,
            )
            return result or {}
        except Exception as e:
            print(f"Sell/wait advice error: {e}")
            return {"error": str(e)}
