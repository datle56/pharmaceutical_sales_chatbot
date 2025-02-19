import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()
# MongoDB connection URI
uri = os.getenv("MONGODB_KEY")

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print("Failed to connect to MongoDB:", e)
    exit()
    
db = client["chatbot"] 
collection = db["products"]

def get_all_products():
    try:
        print("Loading db ....")
        products = list(collection.find())
        return products
    except Exception as e:
        print(f"An error occurred: {e}")
        return []