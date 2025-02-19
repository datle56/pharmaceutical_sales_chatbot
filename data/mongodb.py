import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from utils.embedding import get_embeddings
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()
# MongoDB connection URI
uri = os.getenv("MONGODB_KEY")

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client["chatbot"] 
collection = db["products"]  
embeddings = get_embeddings()
folder_path = "./product"  


# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print("Failed to connect to MongoDB:", e)
    exit()


# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # Load JSON data

                # Generate vector for product description
                if "description" in data and isinstance(data["description"], str):
                    vector = embeddings.embed_query(data["description"])
                    data["vector"] = vector  # Add vector to the document

                # Insert product data into MongoDB
                if isinstance(data, list):  # Nếu file JSON chứa một danh sách sản phẩm
                    for product in data:
                        if "description" in product and isinstance(product["description"], str):
                            product["vector"] = embeddings.embed_query(product["description"])
                    collection.insert_many(data)
                else:  # Nếu file JSON chứa một sản phẩm đơn
                    collection.insert_one(data)

                print(f"Successfully processed {filename}")
        except Exception as e:
            print(f"Error reading or inserting data from {filename}: {e}")



print("Processing complete.")