import os
from dotenv import load_dotenv
from pymongo import MongoClient

#IMPORTANT: load environment variables before import so that langchain modules can access it
load_dotenv()


client = MongoClient(
    os.getenv("MONGODB_URI")
)
collection = client[os.getenv("MONGODB_DATABASE")][os.getenv("MONGODB_COLLECTION")]

collection.delete_many({})