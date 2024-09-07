from langchain_community.llms import DeepInfra
import pickle
import numpy as np
from dotenv import load_dotenv
from functools import lru_cache
import os
import requests
from fastapi import HTTPException
from dotenv import load_dotenv
load_dotenv()

os.getenv('DEEPINFRA_API_TOKEN')


models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

np.random.seed(42)

llm = DeepInfra(model_id="meta-llama/Meta-Llama-3.1-70B-Instruct")
llm.model_kwargs = {
    "temperature": 0.2, 
    "repetition_penalty": 1.2,
    "max_new_tokens": 500,
    "top_p": 0.9,
}

import re

class APIMonitoringAgent:
    def __init__(self, reg_model, clf_model, llm):
        self.reg_model = reg_model
        self.clf_model = clf_model
        self.llm = llm

    @lru_cache(maxsize=None)
    def analyze_metrics(self, traffic_count, error_rate, uptime, cpu_usage, memory_usage, disk_io, concurrent_users):
        try:
            response_time_prediction = self.reg_model.predict([[traffic_count, error_rate, uptime, cpu_usage, memory_usage, disk_io, concurrent_users]])
            response_time_prediction = float(response_time_prediction[0])

            error_prediction = self.clf_model.predict([[response_time_prediction, traffic_count, uptime, cpu_usage, memory_usage, disk_io, concurrent_users]])
            error_prediction = int(error_prediction[0])

            error_confidence = max(self.clf_model.predict_proba([[response_time_prediction, traffic_count, uptime, cpu_usage, memory_usage, disk_io, concurrent_users]])[0])

            risk_level = "High" if response_time_prediction > 500 or error_prediction == 1 else "Low"

            llm_prompt = f"""
            You are an AI agent that monitors API metrics and predicts potential issues.
            Based on the following metrics:

            - Traffic Count: {traffic_count}
            - Error Rate: {error_rate}%
            - Uptime Status: {uptime}%
            - CPU Usage: {cpu_usage}%
            - Memory Usage: {memory_usage}%
            - Disk I/O: {disk_io}
            - Concurrent Users: {concurrent_users}
            - Predicted Response Time: {response_time_prediction}ms
            - Predicted Error Occurrence: {'Yes' if error_prediction == 1 else 'No'}
            - Confidence in Error Prediction: {error_confidence:.2f}
            - Risk Level: {risk_level}

            Please estimate the time to impact and provide a brief recommendation in under 3 lines based on this analysis. Respond in the following format:

            ### Estimated Time to Impact: [your estimation]
            ### Recommendation: [your recommendation]
            """

            llm_response = self.llm.invoke(llm_prompt)
            extracted_response = self.extract_response(llm_response)

            time_to_impact_match = re.search(r"### Estimated Time to Impact: (.+)", extracted_response)
            recommendation_match = re.search(r"### Recommendation: (.+)", extracted_response)

            time_to_impact = time_to_impact_match.group(1) if time_to_impact_match else "Unknown"
            recommendation = recommendation_match.group(1) if recommendation_match else "Unable to provide recommendation."

        except Exception as e:
            print(f"Error processing metrics or LLM response: {e}")
            time_to_impact = "Unknown"
            recommendation = "Error in LLM processing."

        return {
            "predicted_response_time": response_time_prediction,
            "predicted_error_occurrence": error_prediction,
            "error_confidence": error_confidence,
            "risk_level": risk_level,
            "time_to_impact": time_to_impact,
            "recommendation": recommendation,
        }
    
    def extract_response(self, response):
        lines = response.splitlines()
        extracted_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("###") or line.startswith("-"):
                extracted_lines.append(line)
        return "\n".join(extracted_lines)

def fetch_metrics_from_api(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        api_data = response.json()

        return {
            "traffic_count": api_data.get("traffic_count", 0),
            "error_rate": api_data.get("error_rate", 0),
            "uptime": api_data.get("uptime", 0),
            "cpu_usage": api_data.get("cpu_usage", 0.0),
            "memory_usage": api_data.get("memory_usage", 0.0),
            "disk_io": api_data.get("disk_io", 0.0),
            "concurrent_users": api_data.get("concurrent_users", 0)
        }
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching API data: {str(e)}")


with open(os.path.join(models_dir, 'regression_model.pkl'), 'rb') as f:
    reg_model = pickle.load(f)

with open(os.path.join(models_dir, 'classification_model.pkl'), 'rb') as f:
    clf_model = pickle.load(f)

api_agent = APIMonitoringAgent(reg_model, clf_model, llm)