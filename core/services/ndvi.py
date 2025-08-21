import sys
import os
from typing import Optional, Dict, Any

from backend.app.tools.sentinel_cached import ndvi_snapshot_cached, ndvi_quicklook_cached, sentinel_summary_cached
from ..models.domain import NDVIInfo

class NDVIService:
    """Wrapper around existing NDVI/Sentinel tools with comprehensive satellite analysis."""
    
    def __init__(self):
        pass
    
    async def get_ndvi_snapshot(self, 
                               lat: float, 
                               lon: float,
                               aoi_km: Optional[float] = None) -> Dict[str, Any]:
        """Get NDVI snapshot for coordinates."""
        try:
            result = await ndvi_snapshot_cached(
                lat=lat,
                lon=lon,
                aoi_km=aoi_km if aoi_km is not None else 1.0,
            )
            return result or {}
        except Exception as e:
            print(f"NDVI snapshot error: {e}")
            return {"error": str(e)}
    
    async def get_ndvi_quicklook(self,
                                lat: float,
                                lon: float, 
                                aoi_km: Optional[float] = None,
                                debug: bool = False) -> Dict[str, Any]:
        """Get NDVI quicklook image for coordinates."""
        try:
            result = await ndvi_quicklook_cached(
                lat=lat,
                lon=lon,
                aoi_km=aoi_km if aoi_km is not None else 1.0
            )
            if isinstance(result, (bytes, bytearray, memoryview)):
                return {"image_data": result}
            return result or {}
        except Exception as e:
            print(f"NDVI quicklook error: {e}")
            return {"error": str(e)}
    
    async def get_comprehensive_analysis(self,
                                       lat: float,
                                       lon: float,
                                       farm_size_meters: int = 200,
                                       recent_days: int = 20,
                                       debug: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive satellite analysis including NDVI, NDMI, NDWI, LAI 
        with farmer-friendly advice and irrigation recommendations.
        """
        try:
            result = await sentinel_summary_cached(
                lat=lat,
                lon=lon,
                farm_size_meters=farm_size_meters,
                recent_days=recent_days,
                debug=debug
            )
            return result or {}
        except Exception as e:
            print(f"Comprehensive satellite analysis error: {e}")
            return {"error": str(e)}
