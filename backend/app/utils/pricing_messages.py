"""
Enhanced messaging for pricing decisions to make them more intuitive.
"""

def explain_pricing_decision(decision: str, current_price: float, expected_price: float, 
                           trend_slope: float, confidence: float) -> str:
    """
    Create intuitive explanation for pricing decisions.
    """
    price_diff = expected_price - current_price
    price_diff_pct = (price_diff / current_price) * 100 if current_price > 0 else 0
    
    if decision == "WAIT":
        if price_diff > 0:
            return (f"**Wait to sell** - Price expected to rise by ₹{price_diff:.0f} "
                   f"({price_diff_pct:+.1f}%) in the coming days. "
                   f"Current trend is {'upward' if trend_slope > 0 else 'stable'}.")
        else:
            return (f"**Wait to sell** - While expected price is similar, "
                   f"market conditions suggest waiting for better opportunities.")
    
    else:  # SELL_NOW
        if price_diff > 0:
            if price_diff < 50:  # Small increase
                return (f"**Sell now** - Price may rise slightly (₹{price_diff:.0f}), "
                       f"but current price of ₹{current_price:.0f} is good. "
                       f"Secure your earnings now to avoid market volatility.")
            else:
                return (f"**Consider selling** - Expected price rise of ₹{price_diff:.0f} "
                       f"({price_diff_pct:+.1f}%), but current price is reasonable. "
                       f"Decide based on your risk tolerance.")
        elif price_diff < -20:  # Significant decrease expected
            return (f"**Sell immediately** - Price expected to drop by ₹{abs(price_diff):.0f} "
                   f"({price_diff_pct:.1f}%). Current price of ₹{current_price:.0f} "
                   f"is likely the best you'll get soon.")
        else:
            return (f"**Sell now** - Price is stable around ₹{current_price:.0f}. "
                   f"Good time to secure current rates before any market changes.")

def format_price_summary(price_data: dict, advice_data: dict) -> str:
    """Format pricing information in a user-friendly way."""
    if not price_data or not advice_data:
        return "Price information not available."
    
    current = price_data.get("modal_price_inr_per_qtl")
    commodity = price_data.get("commodity", "crop")
    market = price_data.get("market", "local market")
    
    decision = advice_data.get("decision")
    expected = advice_data.get("expected_p50_h")
    confidence = advice_data.get("confidence", 0.7)
    notes = advice_data.get("notes", {})
    trend_slope = notes.get("trend_slope_inr_per_day", 0)
    
    if not current or not expected:
        return f"Current {commodity} price at {market}: ₹{current or 'N/A'}/qtl"
    
    explanation = explain_pricing_decision(decision, current, expected, trend_slope, confidence)
    
    forecast_line = f"\n**Forecast:** Today ₹{current}, expected ₹{expected:.0f}"
    
    return f"{explanation}{forecast_line}"
