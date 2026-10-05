import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


class SimulationDatabase:

    def __init__(self):
        uri = os.getenv(
            "MONGODB_URI",
            "mongodb://localhost:27017/"
        )

        client = MongoClient(uri)

        db = client["ai_agent_iot"]

        self.collection = db["simulations"]

    def find_similar_scenarios(
        self,
        elevation,
        velocity,
        modulation,
        limit=20
    ):

        query = {
            "modulation": modulation,

            "elevation_deg": {
                "$gte": elevation - 2,
                "$lte": elevation + 2
            },

            "velocity_kms": {
                "$gte": velocity - 1,
                "$lte": velocity + 1
            }
        }

        results = self.collection.find(
            query,
            {"_id": 0}
        ).limit(limit)

        return list(results)