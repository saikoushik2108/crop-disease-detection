# API Requirements for Runtime Context

To provide multi-context aware inference, the system requires external API integrations to fetch real-time environmental data. 

## Required API Integrations

| API Name | Purpose | Required? | Free/Paid | Env Variable | Example Data |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OpenWeatherMap** | Current Weather & Temp | Yes | Free Tier | `WEATHER_API_KEY` | Temp, Humidity, Rain, Condition |
| **OpenStreetMap (Nominatim)** | Reverse Geocoding | Yes | Free (No Key) | N/A | City, Region, Country |
| **Tomorrow.io** | High-res Agro-weather | Optional | Free Tier | `TOMORROW_API_KEY` | Soil Moisture, Precipitation |

## Data Mapping to Application Layer

### 1. Environmental Context
The following values will be fetched from the Weather API and displayed in the "Environmental Context" panel:
- **Temperature**: current temperature (°C)
- **Humidity**: relative humidity (%)
- **Weather Condition**: (e.g., "Rain", "Clear", "Overcast")
- **Wind Speed**: current wind speed (m/s)

### 2. Geographic Context
The following values will be fetched via reverse geocoding from Latitude/Longitude:
- **City/Village**: Precise location name
- **Region/State**: Administrative area
- **Country**: Country name

## API Credentials Request
**Attention**: I cannot invent API keys. To activate these features, please provide the following:
1. **OpenWeatherMap API Key** (preferred) OR an alternative weather API key.

If no key is provided, the system will function in "Core ML Mode", providing disease predictions without the environmental context panels.
