def analyze_weather_data(data, analysis_type):
    """
    Analyzes weather data and returns the result as a dictionary.
    :param data: A list of dictionaries containing weather data.
    :param analysis_type: The type of analysis to perform. Must be one of 'average', 'max', 'min', or 'trend'.
    :return: The result of the analysis.
    """

    # Check if the parameters are of the correct type
    if not isinstance(data, list):
        raise TypeError("The data parameter must be a list!")
    if not isinstance(analysis_type, str):
        raise TypeError("The analysis_type parameter must be a string!")
    if analysis_type not in ["average", "max", "min", "trend"]:
        raise ValueError("The analysis_type parameter must be one of 'average', 'max', 'min', or 'trend'!")
    if not data:
        raise ValueError("The data parameter must not be empty!")
    
    # Check if all dictionaries have the required keys
    required_keys = {"date", "temperature", "humidity", "wind_speed"}
    for item in data:
        if not all(key in item for key in required_keys):
            raise ValueError("The data parameter must be a list of dictionaries with the keys 'date', 'temperature', 'humidity', and 'wind_speed'!")
    
    if analysis_type == "average":
        total_temp = sum(item['temperature'] for item in data)
        total_humidity = sum(item['humidity'] for item in data)
        avg_temp = total_temp / len(data)
        avg_humidity = total_humidity / len(data)
        return {"average_temperature": avg_temp, "average_humidity": avg_humidity}

    elif analysis_type == "max":
        max_temp_day = max(data, key=lambda x: x['temperature'])
        return {"max_temperature_date": max_temp_day['date']}

    elif analysis_type == "min":
        min_temp_day = min(data, key=lambda x: x['temperature'])
        return {"min_temperature_date": min_temp_day['date']}

    elif analysis_type == "trend":
        temperatures = [item['temperature'] for item in data]
        if all(temperatures[i] <= temperatures[i + 1] for i in range(len(temperatures) - 1)):
            return "Increasing trend"
        elif all(temperatures[i] >= temperatures[i + 1] for i in range(len(temperatures) - 1)):
            return "Decreasing trend"
        else:
            return "Stable or mixed trend"

    else:
        return "Invalid analysis type"


if __name__ == "__main__":
    # Example usage
    weather_data = [
        {"date": "2023-11-01", "temperature": 20, "humidity": 50, "wind_speed": 5},
        {"date": "2023-11-02", "temperature": 22, "humidity": 45, "wind_speed": 7},
        {"date": "2023-11-03", "temperature": 21, "humidity": 55, "wind_speed": 4},
        # ... add more data as needed
    ]

    result = analyze_weather_data(weather_data, "average")
    print(result)
