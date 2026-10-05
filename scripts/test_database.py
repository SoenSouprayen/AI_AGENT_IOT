from pymongo import MongoClient


client = MongoClient("mongodb://localhost:27017/")

db = client["ai_agent_iot"]

collection = db["simulations"]


print(
    "Nombre de simulations :",
    collection.count_documents({})
)


example = collection.find_one(
    {},
    {"_id": 0}
)

print(example)