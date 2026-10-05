import os
import gdown
from dotenv import load_dotenv

load_dotenv()

file_id = os.getenv("DATASET_GDRIVE_ID")
output_path = os.getenv("DATASET_PATH")

if not file_id:
    raise ValueError("DATASET_GDRIVE_ID absent du fichier .env")

if not output_path:
    raise ValueError("DATASET_PATH absent du fichier .env")

os.makedirs(os.path.dirname(output_path), exist_ok=True)

if os.path.exists(output_path):
    print(f"Dataset déjà présent : {output_path}")
else:
    print("Téléchargement du dataset depuis Google Drive...")

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(
        url=url,
        output=output_path,
        quiet=False
    )

    print(f"Dataset téléchargé : {output_path}")