import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from geopy.geocoders import Nominatim
import httpx
from dotenv import load_dotenv
import google.generativeai as genai

from mcp.server.fastmcp import FastMCP

from pathlib import Path
import logging

# Load environment variables
load_dotenv()

# Ensure logs directory exists
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logging to both console and file
file_handler = logging.FileHandler(LOG_DIR / "mcp_server.log")
console_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[file_handler, console_handler],
)

logger = logging.getLogger(__name__)

# Initialize Gemini AI (safe no-op if key missing)
GENIE_KEY = os.getenv("GEMINI_API_KEY")
if GENIE_KEY:
    genai.configure(api_key=GENIE_KEY)
else:
    logger.warning("GEMINI_API_KEY not set; generate_travel_insights will return an error.")  # FIXED: clearer warning

# Initialize MCP Server
# mcp = FastMCP("TravelPlannerMCP")
# Create an MCP server
mcp = FastMCP(
    name="TravelPlannerMCP",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8000,  # only used for SSE transport (set this to any port)
)
 
# Configuration
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
    AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")
    EVENTBRITE_API_KEY = os.getenv("EVENTBRITE_API_KEY")
    
    # Free API endpoints
    OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
    REST_COUNTRIES_URL = "https://restcountries.com/v3.1"
    EXCHANGE_RATES_URL = "https://api.exchangerate-api.com/v4/latest"
    
    # Geocoding
    NOMINATIM_USER_AGENT = "TravelPlannerMCP/1.0"

# Data Models
@dataclass
class Location:
    name: str
    latitude: float
    longitude: float
    country: str
    timezone: Optional[str] = None

@dataclass
class WeatherData:
    temperature: float
    feels_like: float
    humidity: int
    description: str
    wind_speed: float
    date: str

@dataclass
class FlightOption:
    airline: str
    departure_time: str
    arrival_time: str
    duration: str
    price: float
    currency: str
    stops: int

@dataclass
class AccommodationOption:
    name: str
    type: str  # hotel, hostel, apartment
    price_per_night: float
    rating: float
    address: str
    amenities: List[str]

@dataclass
class ActivityOption:
    name: str
    description: str
    category: str
    price: float
    duration: str
    rating: float
    location: str

# Utilities
class GeocodingService:
    def __init__(self):
        # Keep synchronous geopy client but call it from executor to avoid blocking the loop.
        self.geocoder = Nominatim(user_agent=Config.NOMINATIM_USER_AGENT)

    def _geocode_sync(self, location_name: str):
        # synchronous call to be executed in threadpool
        try:
            return self.geocoder.geocode(location_name)
        except Exception as e:
            # Synchronous handler: rethrow so outer async can log
            raise RuntimeError(f"Geocoding sync error: {e}")

    async def geocode(self, location_name: str) -> Optional[Location]:
        """Non-blocking geocode wrapper using run_in_executor."""
        loop = asyncio.get_running_loop()
        try:
            location = await loop.run_in_executor(None, self._geocode_sync, location_name)  # FIXED: non-blocking
            if location:
                return Location(
                    name=location.address,
                    latitude=location.latitude,
                    longitude=location.longitude,
                    country=self.extract_country(location.address)
                )
        except Exception as e:
            logger.error(f"Geocoding error (async): {e}")
        return None
    
    def extract_country(self, address: str) -> str:
        # Simple country extraction from address (best-effort)
        parts = address.split(", ")
        return parts[-1] if parts else "Unknown"

geocoding_service = GeocodingService()

class WeatherService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)

    async def get_weather(self, location: Location, date: Optional[str] = None) -> WeatherData:
        """Get weather data for a location using NWS API."""
        NWS_API_BASE = "https://api.weather.gov"
        headers = {
            "User-Agent": "TravelPlannerMCP/1.0",
            "Accept": "application/geo+json"
        }
        try:
            # Step 1: get forecast URL for this lat/lon
            points_url = f"{NWS_API_BASE}/points/{location.latitude},{location.longitude}"
            points_resp = await self.client.get(points_url, headers=headers)
            points_resp.raise_for_status()
            points_data = points_resp.json()
            forecast_url = points_data["properties"].get("forecast")
            if not forecast_url:
                raise RuntimeError("Forecast URL not available")

            # Step 2: fetch forecast data
            forecast_resp = await self.client.get(forecast_url, headers=headers)
            forecast_resp.raise_for_status()
            forecast_data = forecast_resp.json()
            periods = forecast_data["properties"].get("periods", [])
            if not periods:
                raise RuntimeError("No forecast periods returned")

            # Step 3: choose a period
            period = periods[0]
            if date:
                for p in periods:
                    if date in p.get("startTime", ""):
                        period = p
                        break

            # Step 4: map to WeatherData
            return WeatherData(
                temperature=period.get("temperature", 0.0),
                feels_like=period.get("temperature", 0.0),  # NWS doesn’t provide feels_like
                humidity=0,  # NWS doesn’t provide humidity in this endpoint
                description=period.get("detailedForecast", ""),
                wind_speed=float(period.get("windSpeed", "0").split()[0]),
                date=period.get("startTime", date or datetime.now().isoformat())
            )

        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return await self._mock_weather_data(location, date)

    async def _mock_weather_data(self, location: Location, date: Optional[str] = None) -> WeatherData:
        return WeatherData(
            temperature=22.5,
            feels_like=25.0,
            humidity=65,
            description="partly cloudy",
            wind_speed=5.2,
            date=date or datetime.now().isoformat()
        )

    async def close(self):
        await self.client.aclose()

class FlightService:
    def __init__(self):
        self.api_key = Config.AMADEUS_API_KEY
        self.api_secret = Config.AMADEUS_API_SECRET
        self.client = httpx.AsyncClient(timeout=30.0)
        self.base_url = "https://test.api.amadeus.com/v2"
        self.token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        self.access_token = None

    async def get_access_token(self):
        """Get OAuth2 access token from Amadeus"""
        if self.access_token:
            return self.access_token
        
        try:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.api_secret
            }
            response = await self.client.post(self.token_url, headers=headers, data=data)
            response.raise_for_status()
            token_data = response.json()
            self.access_token = token_data["access_token"]
            return self.access_token
        except Exception as e:
            logger.error(f"Failed to get Amadeus access token: {e}")
            return None

    def _get_airport_code(self, city_or_code: str) -> str:
        """Convert city names to IATA airport codes"""
        city_to_airport = {
            "chennai": "MAA",
            "paris": "CDG",
            "london": "LHR",
            "new york": "JFK",
            "mumbai": "BOM",
            "delhi": "DEL",
            "bangalore": "BLR",
            "madrid": "MAD",
            "rome": "FCO",
            "amsterdam": "AMS",
            "dubai": "DXB",
            "singapore": "SIN",
            "bangkok": "BKK",
            "tokyo": "NRT",
            "sydney": "SYD"
        }
        
        city_lower = city_or_code.lower().strip()
        return city_to_airport.get(city_lower, city_or_code.upper())

    def _normalize_date(self, date_str: str) -> str:
        # Converts "July 10th, 2025" or other messy formats into YYYY-MM-DD
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date().isoformat()
        except ValueError:
            # fallback: try flexible parsing (needs dateutil if installed)
            from dateutil import parser
            return parser.parse(date_str).date().isoformat()

    async def search_flights(
        self, 
        origin: str, 
        destination: str, 
        departure_date: str,
        budget_max: float = None
    ) -> List[FlightOption]:
        """Search for flights using Amadeus API"""
        try:
            token = await self.get_access_token()
            if not token:
                return await self._mock_flight_data(origin, destination, departure_date, budget_max)

            # Convert city names to airport codes
            origin_code = self._get_airport_code(origin)
            destination_code = self._get_airport_code(destination)

            headers = {"Authorization": f"Bearer {token}"}
            params = {
                "originLocationCode": origin_code,
                "destinationLocationCode": destination_code,
                "departureDate": self._normalize_date(departure_date),
                "adults": 1,
                "max": 10
            }
            
            if budget_max:
                params["maxPrice"] = int(budget_max)

            url = f"{self.base_url}/shopping/flight-offers"
            response = await self.client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = await response.json()

            flights = []
            for offer in data.get("data", []):
                itinerary = offer["itineraries"][0]
                segments = itinerary["segments"]
                
                # Calculate total duration and stops
                duration = itinerary["duration"]
                stops = len(segments) - 1
                
                # Get first segment for departure info
                first_segment = segments[0]
                last_segment = segments[-1]
                
                # Get airline code
                carrier_code = first_segment["carrierCode"]
                airline_name = data.get("dictionaries", {}).get("carriers", {}).get(carrier_code, carrier_code)
                
                flight = FlightOption(
                    airline=airline_name,
                    departure_time=first_segment["departure"]["at"],
                    arrival_time=last_segment["arrival"]["at"],
                    duration=duration.replace("PT", "").replace("H", "h ").replace("M", "m"),
                    price=float(offer["price"]["total"]),
                    currency=offer["price"]["currency"],
                    stops=stops
                )
                flights.append(flight)

            return sorted(flights, key=lambda x: x.price)

        except httpx.HTTPStatusError as e:
            error_text = await e.response.aread() if e.response else str(e)
            logger.error(f"Amadeus flight search HTTP error: {error_text}")
            return await self._mock_flight_data(origin, destination, departure_date, budget_max)
        
        except Exception as e:
            logger.error(f"Amadeus flight search error: {e}")
            return await self._mock_flight_data(origin, destination, departure_date, budget_max)


    
    async def _mock_flight_data(
        self, 
        origin: str, 
        destination: str, 
        departure_date: str,
        budget_max: float = None
    ) -> List[FlightOption]:
        flights = [
            FlightOption(
                airline="Budget Airways",
                departure_time=f"{departure_date}T08:00:00",
                arrival_time=f"{departure_date}T14:30:00",
                duration="6h 30m",
                price=299.99,
                currency="USD",
                stops=0
            ),
            FlightOption(
                airline="Economy Plus",
                departure_time=f"{departure_date}T12:15:00",
                arrival_time=f"{departure_date}T20:45:00",
                duration="8h 30m",
                price=189.99,
                currency="USD",
                stops=1
            ),
            FlightOption(
                airline="Premium Air",
                departure_time=f"{departure_date}T18:00:00",
                arrival_time=f"{departure_date}T23:30:00",
                duration="5h 30m",
                price=459.99,
                currency="USD",
                stops=0
            )
        ]
        
        if budget_max:
            flights = [f for f in flights if f.price <= budget_max]
        
        return sorted(flights, key=lambda x: x.price)

    async def close(self):
        await self.client.aclose()


class AccommodationService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.base_url = "https://test.api.amadeus.com/v1"
        self.token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        self.access_token = None

    async def get_access_token(self):
        """Get OAuth2 access token from Amadeus"""
        if self.access_token:
            return self.access_token
        
        try:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = {
                "grant_type": "client_credentials",
                "client_id": Config.AMADEUS_API_KEY,
                "client_secret": Config.AMADEUS_API_SECRET
            }
            response = await self.client.post(self.token_url, headers=headers, data=data)
            response.raise_for_status()
            token_data = response.json()
            self.access_token = token_data["access_token"]
            return self.access_token
        except Exception as e:
            logger.error(f"Failed to get Amadeus access token: {e}")
            return None

    async def search_accommodations(
        self,
        location: str,
        checkin_date: str,
        checkout_date: str,
        budget_per_night: float = None,
        guests: int = 2
    ) -> List[AccommodationOption]:
        """Search accommodations using Amadeus API"""
        try:
            # First, get coordinates for the location
            geo_location = await geocoding_service.geocode(location)
            if not geo_location:
                return await self._mock_accommodation_data(location, budget_per_night)

            token = await self.get_access_token()
            if not token:
                return await self._mock_accommodation_data(location, budget_per_night)

            headers = {"Authorization": f"Bearer {token}"}
            
            # Search for hotels by geocode
            params = {
                "latitude": geo_location.latitude,
                "longitude": geo_location.longitude,
                "radius": 20,
                "radiusUnit": "KM"
            }

            url = f"{self.base_url}/reference-data/locations/hotels/by-geocode"
            response = await self.client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            accommodations = []
            for hotel in data.get("data", [])[:10]:  # Limit to 10 hotels
                # Generate mock pricing and amenities since hotel search doesn't include prices
                base_price = 50 + (hash(hotel["hotelId"]) % 200)
                if budget_per_night and base_price > budget_per_night:
                    continue
                    
                # Determine hotel type based on chain code or name
                hotel_name = hotel.get("name", "Unknown Hotel")
                hotel_type = "hotel"
                if any(word in hotel_name.lower() for word in ["hostel", "backpack"]):
                    hotel_type = "hostel"
                    base_price *= 0.4
                elif any(word in hotel_name.lower() for word in ["apartment", "apart", "suite"]):
                    hotel_type = "apartment"
                    base_price *= 0.8

                # Generate rating and amenities
                rating = 3.5 + (hash(hotel["hotelId"]) % 15) / 10
                
                basic_amenities = ["WiFi", "Reception"]
                if rating > 4.0:
                    basic_amenities.extend(["Restaurant", "Room Service", "Concierge"])
                if rating > 4.5:
                    basic_amenities.extend(["Spa", "Gym", "Pool"])
                if hotel_type == "apartment":
                    basic_amenities.extend(["Kitchen", "Washing Machine"])

                accommodation = AccommodationOption(
                    name=hotel_name,
                    type=hotel_type,
                    price_per_night=round(base_price, 2),
                    rating=round(rating, 1),
                    address=hotel.get("address", {}).get("countryCode", location),
                    amenities=basic_amenities
                )
                accommodations.append(accommodation)

            return sorted(accommodations, key=lambda x: x.price_per_night)

        except Exception as e:
            logger.error(f"Amadeus accommodation search error: {e}")
            return await self._mock_accommodation_data(location, budget_per_night)
    
    async def _mock_accommodation_data(
        self,
        location: str,
        budget_per_night: float = None
    ) -> List[AccommodationOption]:
        accommodations = [
            AccommodationOption(
                name="Budget Hostel Central",
                type="hostel",
                price_per_night=25.0,
                rating=4.2,
                address=f"Downtown {location}",
                amenities=["WiFi", "Shared Kitchen", "24h Reception"]
            ),
            AccommodationOption(
                name="Comfort Inn Express",
                type="hotel",
                price_per_night=89.0,
                rating=4.5,
                address=f"Business District, {location}",
                amenities=["WiFi", "Breakfast", "Gym", "Pool"]
            ),
            AccommodationOption(
                name="Luxury Boutique Hotel",
                type="hotel",
                price_per_night=189.0,
                rating=4.8,
                address=f"Historic Center, {location}",
                amenities=["WiFi", "Spa", "Restaurant", "Concierge", "Room Service"]
            ),
            AccommodationOption(
                name="Modern Apartment",
                type="apartment",
                price_per_night=65.0,
                rating=4.6,
                address=f"Residential Area, {location}",
                amenities=["WiFi", "Kitchen", "Washing Machine", "Balcony"]
            )
        ]
        
        if budget_per_night:
            accommodations = [a for a in accommodations if a.price_per_night <= budget_per_night]
        
        return sorted(accommodations, key=lambda x: x.price_per_night)

    async def close(self):
        await self.client.aclose()


class ActivityService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.base_url = "https://test.api.amadeus.com/v1"
        self.token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        self.access_token = None

    async def get_access_token(self):
        """Get OAuth2 access token from Amadeus"""
        if self.access_token:
            return self.access_token
        
        try:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = {
                "grant_type": "client_credentials",
                "client_id": Config.AMADEUS_API_KEY,
                "client_secret": Config.AMADEUS_API_SECRET
            }
            response = await self.client.post(self.token_url, headers=headers, data=data)
            response.raise_for_status()
            token_data = response.json()
            self.access_token = token_data["access_token"]
            return self.access_token
        except Exception as e:
            logger.error(f"Failed to get Amadeus access token: {e}")
            return None

    async def search_activities(
        self,
        location: str,
        category: Optional[str] = None,
        budget_max: float = None
    ) -> List[ActivityOption]:
        """Search activities using Amadeus API"""
        try:
            # First, get coordinates for the location
            geo_location = await geocoding_service.geocode(location)
            if not geo_location:
                return await self._mock_activity_data(location, category, budget_max)

            token = await self.get_access_token()
            if not token:
                return await self._mock_activity_data(location, category, budget_max)

            headers = {"Authorization": f"Bearer {token}"}
            
            params = {
                "latitude": geo_location.latitude,
                "longitude": geo_location.longitude,
                "radius": 10  # 10km radius
            }

            url = f"{self.base_url}/shopping/activities"
            response = await self.client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            activities = []
            for activity_data in data.get("data", []):
                # Map activity to our category system
                activity_name = activity_data.get("name", "Unknown Activity")
                activity_category = self._categorize_activity(activity_name, activity_data.get("shortDescription", ""))
                
                # Filter by category if specified
                if category and activity_category != category:
                    continue
                
                # Get price
                price_info = activity_data.get("price", {})
                price = float(price_info.get("amount", 0))
                
                # Filter by budget
                if budget_max and price > budget_max:
                    continue
                
                # Estimate duration based on activity type
                duration = self._estimate_duration(activity_category, activity_name)
                
                activity = ActivityOption(
                    name=activity_name,
                    description=activity_data.get("shortDescription", ""),
                    category=activity_category,
                    price=price,
                    duration=duration,
                    rating=float(activity_data.get("rating", 4.0)),
                    location=f"{geo_location.name}"
                )
                activities.append(activity)

            return sorted(activities, key=lambda x: x.rating, reverse=True)

        except Exception as e:
            logger.error(f"Amadeus activity search error: {e}")
            return await self._mock_activity_data(location, category, budget_max)

    def _categorize_activity(self, name: str, description: str) -> str:
        """Categorize activity based on name and description"""
        text = (name + " " + description).lower()
        
        if any(word in text for word in ["museum", "gallery", "art", "cultural", "historic", "heritage"]):
            return "culture"
        elif any(word in text for word in ["food", "culinary", "taste", "restaurant", "cooking", "wine"]):
            return "food"
        elif any(word in text for word in ["adventure", "hiking", "climbing", "sports", "outdoor", "zip", "bike"]):
            return "adventure"
        elif any(word in text for word in ["tour", "walking", "city", "sightseeing", "guide"]):
            return "sightseeing"
        else:
            return "culture"

    def _estimate_duration(self, category: str, name: str) -> str:
        """Estimate activity duration based on category and name"""
        if category == "culture":
            return "2-3 hours"
        elif category == "food":
            return "3-4 hours"
        elif category == "adventure":
            return "4-6 hours"
        elif category == "sightseeing":
            if "walking" in name.lower():
                return "2-3 hours"
            return "4-5 hours"
        else:
            return "2-4 hours"
    
    async def _mock_activity_data(
        self,
        location: str,
        category: Optional[str] = None,
        budget_max: float = None
    ) -> List[ActivityOption]:
        activities = [
            ActivityOption(
                name="City Walking Tour",
                description="Explore the historic downtown area with a local guide",
                category="sightseeing",
                price=15.0,
                duration="3 hours",
                rating=4.7,
                location=f"Historic District, {location}"
            ),
            ActivityOption(
                name="Local Food Experience",
                description="Taste authentic local cuisine at traditional restaurants",
                category="food",
                price=45.0,
                duration="4 hours",
                rating=4.8,
                location=f"Food Quarter, {location}"
            ),
            ActivityOption(
                name="Museum Visit",
                description="Visit the main city museum with cultural artifacts",
                category="culture",
                price=12.0,
                duration="2 hours",
                rating=4.4,
                location=f"Cultural District, {location}"
            ),
            ActivityOption(
                name="Adventure Park",
                description="Outdoor activities including hiking and zip-lining",
                category="adventure",
                price=75.0,
                duration="6 hours",
                rating=4.9,
                location=f"Nature Reserve, {location}"
            ),
            ActivityOption(
                name="Art Gallery Tour",
                description="Contemporary and classical art exhibitions",
                category="culture",
                price=20.0,
                duration="2.5 hours",
                rating=4.5,
                location=f"Arts District, {location}"
            )
        ]
        
        if category:
            activities = [a for a in activities if a.category == category]
        
        if budget_max:
            activities = [a for a in activities if a.price <= budget_max]
        
        return sorted(activities, key=lambda x: x.rating, reverse=True)

    async def close(self):
        await self.client.aclose()

class CountryInfoService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)  # FIXED: prepared AsyncClient

    async def get_country_info(self, country_name: str) -> Dict[str, Any]:
        """Get country information using REST Countries API"""
        try:
            url = f"{Config.REST_COUNTRIES_URL}/name/{country_name}"
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            if data:
                country = data[0]
                # safer extraction with defaults and defensive checks
                name = country.get("name", {}).get("common") if country.get("name") else country_name
                capital = country.get("capital", ["Unknown"])[0] if country.get("capital") else "Unknown"
                currencies = list(country.get("currencies", {}).keys()) if country.get("currencies") else []
                currency = currencies[0] if currencies else "USD"
                languages = list(country.get("languages", {}).values()) if country.get("languages") else ["English"]
                timezone = country.get("timezones", ["UTC"])[0] if country.get("timezones") else "UTC"
                region = country.get("region", "Unknown") or "Unknown"
                population = country.get("population", 0) or 0
                return {
                    "name": name,
                    "capital": capital,
                    "currency": currency,
                    "languages": languages,
                    "timezone": timezone,
                    "region": region,
                    "population": population
                }
        except Exception as e:
            logger.error(f"Country info error: {e}")
        
        return {
            "name": country_name,
            "capital": "Unknown",
            "currency": "USD",
            "languages": ["English"],
            "timezone": "UTC",
            "region": "Unknown",
            "population": 0
        }

    async def close(self):
        await self.client.aclose()  # FIXED: client cleanup

# Initialize services
weather_service = WeatherService()
flight_service = FlightService()
accommodation_service = AccommodationService()
activity_service = ActivityService()
country_service = CountryInfoService()

# MCP Tools Implementation

@mcp.tool()
async def search_flights(
    origin: str,
    destination: str, 
    departure_date: str,
    budget_max: Optional[float] = None
) -> List[Dict[str, Any]]:
    try:
        flights = await flight_service.search_flights(origin, destination, departure_date, budget_max)
        return [
            {
                "airline": f.airline,
                "departure_time": f.departure_time,
                "arrival_time": f.arrival_time,
                "duration": f.duration,
                "price": f.price,
                "currency": f.currency,
                "stops": f.stops
            } for f in flights
        ]
    except Exception as e:
        logger.error(f"Flight search error: {e}")
        return []

@mcp.tool()
async def search_accommodations(
    location: str,
    checkin_date: str,
    checkout_date: str,
    budget_per_night: Optional[float] = None,
    guests: int = 2
) -> List[Dict[str, Any]]:
    try:
        accommodations = await accommodation_service.search_accommodations(
            location, checkin_date, checkout_date, budget_per_night, guests
        )
        return [
            {
                "name": a.name,
                "type": a.type,
                "price_per_night": a.price_per_night,
                "rating": a.rating,
                "address": a.address,
                "amenities": a.amenities
            } for a in accommodations
        ]
    except Exception as e:
        logger.error(f"Accommodation search error: {e}")
        return []

@mcp.tool()
async def search_activities(
    location: str,
    category: Optional[str] = None,
    budget_max: Optional[float] = None
) -> List[Dict[str, Any]]:
    try:
        activities = await activity_service.search_activities(location, category, budget_max)
        return [
            {
                "name": a.name,
                "description": a.description,
                "category": a.category,
                "price": a.price,
                "duration": a.duration,
                "rating": a.rating,
                "location": a.location
            } for a in activities
        ]
    except Exception as e:
        logger.error(f"Activity search error: {e}")
        return []

@mcp.tool()
async def get_weather_forecast(
    location: str,
    date: Optional[str] = None
) -> Dict[str, Any]:
    try:
        # Use async geocoding (non-blocking)
        geo_location = await geocoding_service.geocode(location)  # FIXED: await non-blocking geocode
        if not geo_location:
            return {"error": f"Could not find location: {location}"}
        
        weather = await weather_service.get_weather(geo_location, date)
        return {
            "location": geo_location.name,
            "temperature": weather.temperature,
            "feels_like": weather.feels_like,
            "humidity": weather.humidity,
            "description": weather.description,
            "wind_speed": weather.wind_speed,
            "date": weather.date
        }
    except Exception as e:
        logger.error(f"Weather forecast error: {e}")
        return {"error": str(e)}

@mcp.tool()
async def get_location_info(location: str) -> Dict[str, Any]:
    try:
        geo_location = await geocoding_service.geocode(location)  # FIXED: await non-blocking geocode
        if not geo_location:
            return {"error": f"Could not find location: {location}"}
        
        country_info = await country_service.get_country_info(geo_location.country)
        
        return {
            "name": geo_location.name,
            "latitude": geo_location.latitude,
            "longitude": geo_location.longitude,
            "country": geo_location.country,
            "country_info": country_info
        }
    except Exception as e:
        logger.error(f"Location info error: {e}")
        return {"error": str(e)}

@mcp.tool()
async def calculate_trip_budget(
    flights_budget: float,
    accommodation_budget: float,
    activities_budget: float,
    food_budget: float,
    miscellaneous_budget: Optional[float] = None
) -> Dict[str, Any]:
    try:
        misc_budget = miscellaneous_budget or 0.0
        total_budget = flights_budget + accommodation_budget + activities_budget + food_budget + misc_budget
        
        breakdown = {
            "flights": flights_budget,
            "accommodation": accommodation_budget,
            "activities": activities_budget,
            "food": food_budget,
            "miscellaneous": misc_budget,
            "total": total_budget
        }
        
        percentages = {
            category: (amount / total_budget * 100) if total_budget > 0 else 0
            for category, amount in breakdown.items() if category != "total"
        }
        
        return {
            "budget_breakdown": breakdown,
            "percentages": percentages,
            "recommendations": {
                "is_balanced": 20 <= percentages.get("flights", 0) <= 40 and 25 <= percentages.get("accommodation", 0) <= 45,
                "suggestions": [
                    "Consider allocating 30-40% for accommodation",
                    "Reserve 20-30% for flights", 
                    "Budget 15-25% for activities",
                    "Allocate 15-20% for food",
                    "Keep 5-10% for miscellaneous expenses"
                ]
            }
        }
    except Exception as e:
        logger.error(f"Budget calculation error: {e}")
        return {"error": str(e)}

@mcp.tool()
async def generate_travel_insights(
    destination: str,
    travel_dates: str,
    interests: List[str]
) -> Dict[str, Any]:
    try:
        if not Config.GEMINI_API_KEY:
            return {"error": "Gemini API key not configured"}
        
        prompt = f"""
        As a travel expert, provide comprehensive insights for a trip to {destination}
        during {travel_dates}. The traveler is interested in: {', '.join(interests)}.

        Please provide JSON with keys:
        best_time_to_visit, must_see, local_customs, budget_recommendations, safety, transport_tips, food, hidden_gems

        Note: Ensure the response is valid JSON only. No additional text outside the JSON block.
        """
        # Use Gemini (best-effort); response parsing is defensive
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        insights_text = getattr(response, "text", "")  # safe access
        # Try to parse JSON, fallback to raw text
        insights_parsed = None
        try:
            insights_parsed = json.loads(insights_text)
        except Exception:
            # If the model returned JSON-like but with trailing text, attempt to extract a JSON substring
            try:
                # find first and last brace to attempt parsing
                start = insights_text.find("{")
                end = insights_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = insights_text[start:end+1]
                    insights_parsed = json.loads(candidate)
            except Exception:
                insights_parsed = None  # keep fallback below

        return {
            "destination": destination,
            "travel_dates": travel_dates,
            "interests": interests,
            "insights": insights_parsed if insights_parsed is not None else insights_text,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Travel insights generation error: {e}")
        return {"error": str(e)}

# MCP Resources

@mcp.resource("travel://destination/{destination}")
def get_destination_info(destination: str) -> str:
    """Get comprehensive destination information"""
    return f"""# {destination} Travel Guide

## Overview
{destination} is a popular travel destination offering diverse experiences for visitors.

## Key Information
- **Best Time to Visit**: Varies by season and weather patterns
- **Currency**: Check local currency requirements
- **Language**: Local language information available
- **Time Zone**: Consult local time zone information

## Popular Attractions
- Historic landmarks and monuments
- Museums and cultural sites  
- Natural attractions and parks
- Entertainment districts

## Local Tips
- Transportation options
- Cultural customs
- Tipping practices
- Safety considerations

*Use the search tools to get specific, real-time information about flights, accommodation, activities, and weather.*"""

@mcp.resource("travel://weather/{location}/{date}")
def get_weather_resource(location: str, date: str) -> str:
    """Get weather information resource"""
    return f"""# Weather Information for {location}

## Date: {date}

Use the get_weather_forecast tool for real-time weather data including:
- Current temperature
- Weather conditions  
- Humidity levels
- Wind speed
- Extended forecast

*This resource provides the latest weather information to help plan your activities.*"""

if __name__ == "__main__":
    try:
        mcp.run(transport="sse")  # Run the MCP server with SSE transport
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Travel Planner MCP Server stopped by user (CTRL+C).")