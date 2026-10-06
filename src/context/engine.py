import os
import requests
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class ContextEngine:
    """
    The Runtime Context Engine.

    Fetches real-time environmental and geographic data from external APIs.
    Designed to fail gracefully if APIs are unavailable or keys are missing.
    """

    def __init__(self):
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
        self.geocoding_api_url = "https://nominatim.openstreetmap.org/reverse"
        self.weather_api_url = "https://api.openweathermap.org/data/2.5/weather"

    def get_location_context(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Convert coordinates to a human-readable location using Nominatim (OpenStreetMap).
        """
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "format": "json",
                "addressdetails": 1
            }
            # Nominatim requires a User-Agent header to avoid 403s
            headers = {"User-Agent": "PaddyDiseaseAI-ResearchProject/1.0"}
            response = requests.get(self.geocoding_api_url, params=params, headers=headers, timeout=5)

            if response.status_code == 200:
                data = response.json()
                address = data.get("address", {})
                return {
                    "city": address.get("city") or address.get("town") or address.get("village") or "Unknown",
                    "region": address.get("state") or address.get("province") or "Unknown",
                    "country": address.get("country") or "Unknown",
                    "status": "available"
                }
        except Exception as e:
            print(f"Location API Error: {e}")

        return {"status": "unavailable", "error": "Location service failed"}

    def get_weather_context(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch current environmental conditions using OpenWeatherMap.
        """
        if not self.weather_api_key:
            return {"status": "unavailable", "error": "WEATHER_API_KEY missing"}

        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.weather_api_key,
                "units": "metric"
            }
            response = requests.get(self.weather_api_url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                main = data.get("main", {})
                weather = data.get("weather", [{}])[0]

                return {
                    "temperature": main.get("temp"),
                    "humidity": main.get("humidity"),
                    "condition": weather.get("main"),
                    "description": weather.get("description"),
                    "status": "available"
                }
        except Exception as e:
            print(f"Weather API Error: {e}")

        return {"status": "unavailable", "error": "Weather service failed"}

    def get_full_context(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Orchestrate both location and weather retrieval.
        """
        return {
            "location": self.get_location_context(lat, lon),
            "environment": self.get_weather_context(lat, lon)
        }
