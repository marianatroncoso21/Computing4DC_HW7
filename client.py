import json
import requests

API_URL = "http://localhost:8000/predict"
INPUT_FILE = "example_input.json"

def load_patient(path):
    with open(path, "r") as f:
        return json.load(f)


def get_prediction(patient_data):
    try:
        response = requests.post(url=API_URL, json=patient_data, timeout=30)
        response.raise_for_status()
        return response.json()["prediction"]
    except requests.exceptions.ConnectionError as e:
        print(f"Error: Unable to connect to the API. Please check if the server is running.: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error: Unable to decode the JSON response from the API: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the request: {e}")
        return None


if __name__ == "__main__":
    loaded_patient = load_patient(INPUT_FILE)
    prediction = get_prediction(loaded_patient)
    if prediction is not None:
        print(f"Prediction: {prediction}")
    