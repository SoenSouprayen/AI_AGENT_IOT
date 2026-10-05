import os
from dotenv import load_dotenv

load_dotenv()

csv_path = os.getenv("DATASET_PATH")
mongodb_uri = os.getenv("MONGODB_URI")

print(csv_path)