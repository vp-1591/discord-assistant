import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Configuration
STORE_NAME = "My_Project_Knowledge_Base"

def delete_store():
    # 1. Initialize Client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment.")
        return

    client = genai.Client(api_key=api_key)
    
    # 2. Find the Store
    print(f"Looking for store: '{STORE_NAME}'...")
    store_resource_name = None
    try:
        existing_stores = list(client.file_search_stores.list())
        for store in existing_stores:
            if store.display_name == STORE_NAME:
                store_resource_name = store.name
                break
    except Exception as e:
        print(f"Error listing stores: {e}")
        return
            
    if not store_resource_name:
        print(f"Store '{STORE_NAME}' not found. Nothing to delete.")
        return

    print(f"Found store: {store_resource_name}")
    print(f"Deleting the entire store '{STORE_NAME}' and all its contents...")

    # 3. Delete the Store
    try:
        # Use force=True to ensure the store is deleted even if it contains documents (stuck or otherwise)
        client.file_search_stores.delete(
            name=store_resource_name,
            config={'force': True}
        )
        print("Store deleted successfully.")
        print("Note: You will need to create a new store (using create_store.py) before uploading files again.")

    except Exception as e:
        print(f"An error occurred while deleting the store: {e}")

if __name__ == "__main__":
    delete_store()