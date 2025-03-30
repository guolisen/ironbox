#!/usr/bin/env python3
"""
Weather MCP server for IronBox.
"""
import json
import logging
import sys
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherServer:
    """Weather MCP server."""
    
    def __init__(self):
        """Initialize WeatherServer."""
        self.cities = {
            "new york": {
                "country": "United States",
                "latitude": 40.7128,
                "longitude": -74.0060,
            },
            "london": {
                "country": "United Kingdom",
                "latitude": 51.5074,
                "longitude": -0.1278,
            },
            "tokyo": {
                "country": "Japan",
                "latitude": 35.6762,
                "longitude": 139.6503,
            },
            "sydney": {
                "country": "Australia",
                "latitude": -33.8688,
                "longitude": 151.2093,
            },
            "paris": {
                "country": "France",
                "latitude": 48.8566,
                "longitude": 2.3522,
            },
            "beijing": {
                "country": "China",
                "latitude": 39.9042,
                "longitude": 116.4074,
            },
            "moscow": {
                "country": "Russia",
                "latitude": 55.7558,
                "longitude": 37.6173,
            },
            "rio de janeiro": {
                "country": "Brazil",
                "latitude": -22.9068,
                "longitude": -43.1729,
            },
            "cairo": {
                "country": "Egypt",
                "latitude": 30.0444,
                "longitude": 31.2357,
            },
            "mumbai": {
                "country": "India",
                "latitude": 19.0760,
                "longitude": 72.8777,
            },
        }
        
        self.weather_conditions = [
            "Clear",
            "Partly Cloudy",
            "Cloudy",
            "Overcast",
            "Light Rain",
            "Rain",
            "Heavy Rain",
            "Thunderstorm",
            "Snow",
            "Fog",
            "Mist",
            "Haze",
        ]
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an MCP request.
        
        Args:
            request: Request object
            
        Returns:
            Response object
        """
        try:
            method = request.get("method")
            params = request.get("params", {})
            
            if method == "server.info":
                return self._handle_server_info()
            elif method == "tool.list":
                return self._handle_tool_list()
            elif method == "tool.call":
                return await self._handle_tool_call(params)
            elif method == "resource.list":
                return self._handle_resource_list()
            elif method == "resource.listTemplates":
                return self._handle_resource_list_templates()
            elif method == "resource.read":
                return await self._handle_resource_read(params)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}",
                },
            }
    
    def _handle_server_info(self) -> Dict[str, Any]:
        """
        Handle server.info request.
        
        Returns:
            Response object
        """
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "name": "weather",
                "version": "0.1.0",
                "description": "Weather information provider",
                "capabilities": {
                    "tools": True,
                    "resources": True,
                },
            },
        }
    
    def _handle_tool_list(self) -> Dict[str, Any]:
        """
        Handle tool.list request.
        
        Returns:
            Response object
        """
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "get_current_weather",
                        "description": "Get current weather for a city",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "City name",
                                },
                                "units": {
                                    "type": "string",
                                    "description": "Temperature units (celsius or fahrenheit)",
                                    "enum": ["celsius", "fahrenheit"],
                                    "default": "celsius",
                                },
                            },
                            "required": ["city"],
                        },
                    },
                    {
                        "name": "get_forecast",
                        "description": "Get weather forecast for a city",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "City name",
                                },
                                "days": {
                                    "type": "integer",
                                    "description": "Number of days (1-7)",
                                    "minimum": 1,
                                    "maximum": 7,
                                    "default": 3,
                                },
                                "units": {
                                    "type": "string",
                                    "description": "Temperature units (celsius or fahrenheit)",
                                    "enum": ["celsius", "fahrenheit"],
                                    "default": "celsius",
                                },
                            },
                            "required": ["city"],
                        },
                    },
                ],
            },
        }
    
    async def _handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle tool.call request.
        
        Args:
            params: Request parameters
            
        Returns:
            Response object
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "get_current_weather":
            return await self._handle_get_current_weather(arguments)
        elif tool_name == "get_forecast":
            return await self._handle_get_forecast(arguments)
        else:
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {
                    "code": -32601,
                    "message": f"Tool not found: {tool_name}",
                },
            }
    
    async def _handle_get_current_weather(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle get_current_weather tool call.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Response object
        """
        city = arguments.get("city", "").lower()
        units = arguments.get("units", "celsius")
        
        if not city:
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "isError": True,
                    "content": [
                        {
                            "type": "text",
                            "text": "City is required",
                        },
                    ],
                },
            }
        
        # Check if city exists
        if city not in self.cities:
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "isError": True,
                    "content": [
                        {
                            "type": "text",
                            "text": f"City not found: {city}",
                        },
                    ],
                },
            }
        
        # Generate random weather data
        weather = self._generate_weather(city, units)
        
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(weather, indent=2),
                    },
                ],
            },
        }
    
    async def _handle_get_forecast(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle get_forecast tool call.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Response object
        """
        city = arguments.get("city", "").lower()
        days = min(max(arguments.get("days", 3), 1), 7)
        units = arguments.get("units", "celsius")
        
        if not city:
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "isError": True,
                    "content": [
                        {
                            "type": "text",
                            "text": "City is required",
                        },
                    ],
                },
            }
        
        # Check if city exists
        if city not in self.cities:
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "isError": True,
                    "content": [
                        {
                            "type": "text",
                            "text": f"City not found: {city}",
                        },
                    ],
                },
            }
        
        # Generate random forecast data
        forecast = []
        for i in range(days):
            date = datetime.now() + timedelta(days=i)
            weather = self._generate_weather(city, units, date)
            forecast.append(weather)
        
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(forecast, indent=2),
                    },
                ],
            },
        }
    
    def _handle_resource_list(self) -> Dict[str, Any]:
        """
        Handle resource.list request.
        
        Returns:
            Response object
        """
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "resources": [
                    {
                        "uri": "weather://cities",
                        "name": "List of supported cities",
                        "description": "List of cities supported by the weather service",
                        "mimeType": "application/json",
                    },
                ],
            },
        }
    
    def _handle_resource_list_templates(self) -> Dict[str, Any]:
        """
        Handle resource.listTemplates request.
        
        Returns:
            Response object
        """
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "resourceTemplates": [
                    {
                        "uriTemplate": "weather://{city}/current",
                        "name": "Current weather for a city",
                        "description": "Get current weather for a specific city",
                        "mimeType": "application/json",
                    },
                    {
                        "uriTemplate": "weather://{city}/forecast/{days}",
                        "name": "Weather forecast for a city",
                        "description": "Get weather forecast for a specific city",
                        "mimeType": "application/json",
                    },
                ],
            },
        }
    
    async def _handle_resource_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle resource.read request.
        
        Args:
            params: Request parameters
            
        Returns:
            Response object
        """
        uri = params.get("uri")
        
        if not uri:
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {
                    "code": -32602,
                    "message": "URI is required",
                },
            }
        
        # Handle static resources
        if uri == "weather://cities":
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(list(self.cities.keys()), indent=2),
                        },
                    ],
                },
            }
        
        # Handle resource templates
        import re
        
        # Current weather
        current_match = re.match(r"weather://([^/]+)/current", uri)
        if current_match:
            city = current_match.group(1).lower()
            
            if city not in self.cities:
                return {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {
                        "code": -32602,
                        "message": f"City not found: {city}",
                    },
                }
            
            weather = self._generate_weather(city)
            
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(weather, indent=2),
                        },
                    ],
                },
            }
        
        # Forecast
        forecast_match = re.match(r"weather://([^/]+)/forecast/(\d+)", uri)
        if forecast_match:
            city = forecast_match.group(1).lower()
            days = min(max(int(forecast_match.group(2)), 1), 7)
            
            if city not in self.cities:
                return {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {
                        "code": -32602,
                        "message": f"City not found: {city}",
                    },
                }
            
            forecast = []
            for i in range(days):
                date = datetime.now() + timedelta(days=i)
                weather = self._generate_weather(city, "celsius", date)
                forecast.append(weather)
            
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(forecast, indent=2),
                        },
                    ],
                },
            }
        
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32602,
                "message": f"Invalid URI: {uri}",
            },
        }
    
    def _generate_weather(self, city: str, units: str = "celsius", date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate random weather data for a city.
        
        Args:
            city: City name
            units: Temperature units
            date: Optional date
            
        Returns:
            Weather data
        """
        city_info = self.cities.get(city, {})
        date = date or datetime.now()
        
        # Generate random temperature based on latitude and season
        latitude = city_info.get("latitude", 0)
        month = date.month
        
        # Northern hemisphere: summer = June-August, winter = December-February
        # Southern hemisphere: summer = December-February, winter = June-August
        is_northern = latitude >= 0
        is_summer_month = (month >= 6 and month <= 8) if is_northern else (month == 12 or month <= 2)
        is_winter_month = (month == 12 or month <= 2) if is_northern else (month >= 6 and month <= 8)
        
        # Base temperature range
        if is_summer_month:
            base_temp = random.uniform(20, 35)
        elif is_winter_month:
            base_temp = random.uniform(-10, 15)
        else:
            base_temp = random.uniform(10, 25)
        
        # Adjust for latitude
        temp_celsius = base_temp - abs(latitude) * 0.3
        
        # Convert to fahrenheit if needed
        temp = temp_celsius if units == "celsius" else temp_celsius * 9/5 + 32
        
        # Generate random weather condition
        condition = random.choice(self.weather_conditions)
        
        # Generate random humidity
        humidity = random.randint(30, 90)
        
        # Generate random wind speed
        wind_speed = random.uniform(0, 20)
        
        # Generate random precipitation
        precipitation = 0
        if condition in ["Light Rain", "Rain", "Heavy Rain", "Thunderstorm", "Snow"]:
            precipitation = random.uniform(0.1, 30)
        
        return {
            "city": city.title(),
            "country": city_info.get("country"),
            "date": date.strftime("%Y-%m-%d"),
            "time": date.strftime("%H:%M:%S"),
            "condition": condition,
            "temperature": round(temp, 1),
            "units": units,
            "humidity": humidity,
            "wind_speed": round(wind_speed, 1),
            "precipitation": round(precipitation, 1),
            "latitude": city_info.get("latitude"),
            "longitude": city_info.get("longitude"),
        }


async def main():
    """Main function."""
    server = WeatherServer()
    
    # Read from stdin
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = await server.handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {line}")
        except Exception as e:
            logger.error(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
