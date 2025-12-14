import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Note: You must have the 'google-genai' library installed: pip install google-genai
# Ensure your GEMINI_API_KEY environment variable is set.

# --- Configuration ---
STORE_NAME = "My_Project_Knowledge_Base"
# ---------------------

def create_file_search_store():
    """
    Creates a new File Search Store using the Gemini API client.
    """
    try:
        # 1. Initialize the client (automatically uses the GOOGLE_API_KEY env variable)
        client = genai.Client()

        print(f"Checking for existing store with display name: '{STORE_NAME}'...")

        # 2. Check if a store with this display name already exists (optional but good practice)
        existing_store = None
        for store in client.file_search_stores.list():
            if store.display_name == STORE_NAME:
                existing_store = store
                break

        if existing_store:
            print(f"Store already exists: {existing_store.name}")
            return existing_store
        
        # 3. Create the File Search Store
        print(f"Store not found. Creating new store: '{STORE_NAME}'...")
        
        file_search_store = client.file_search_stores.create(
            config={"display_name": STORE_NAME}
        )
        
        print(f"✅ Successfully created store!")
        print(f"Store Name (Resource ID): {file_search_store.name}")
        print("-" * 30)
        print("NEXT STEP: Use the Resource ID (fileSearchStores/...) to upload documents.")
        
        return file_search_store

    except Exception as e:
        print(f"An error occurred during store creation: {e}")
        return None

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: Please set the GEMINI_API_KEY environment variable.")
    else:
        create_file_search_store()