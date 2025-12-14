import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Configuration
STORE_NAME = "My_Project_Knowledge_Base"
FILE_PREFIX = "discord_chat_part_"
FILE_EXTENSION = ".txt"
START_INDEX = 1
END_INDEX = 10

def upload_chunks():
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
        print(f"Store '{STORE_NAME}' not found.")
        return

    print(f"Found store: {store_resource_name}")

    # 3. Iterate through files
    for i in range(START_INDEX, END_INDEX + 1):
        file_path = f"{FILE_PREFIX}{i}{FILE_EXTENSION}"
        
        if not os.path.exists(file_path):
            print(f"[{i}/{END_INDEX}] File {file_path} not found. Skipping.")
            continue

        print(f"\n[{i}/{END_INDEX}] Processing {file_path}...")
        uploaded_file_resource = None

        try:
            # Step 3a: Upload to Files API
            print(f"  - Uploading to Gemini Files API...")
            file_upload_config = {
                "mime_type": "text/plain",
                "display_name": file_path
            }
            uploaded_file_resource = client.files.upload(
                file=file_path,
                config=file_upload_config
            )
            print(f"  - Upload success: {uploaded_file_resource.name}")

            # Step 3b: Import to Store
            print(f"  - Importing into store...")
            
            # Note: Using keyword arguments that likely map to the proto/client definition.
            # If 'file_search_store_name' fails, we might need positional args or 'name'.
            # Based on previous interactions, we try this specific kwargs pattern used in other successful snippets or standard google genai.
            operation = client.file_search_stores.import_file(
                file_search_store_name=store_resource_name,
                file_name=uploaded_file_resource.name
            )
            
            # 4. Poll for completion
            while not operation.done:
                print("  - Indexing...", end="\r")
                time.sleep(2)
                operation = client.operations.get(operation)
            
            if operation.error:
                print(f"\n  - Error during indexing: {operation.error}")
            else:
                print(f"\n  - Success! Chunk {i} indexed.")

        except Exception as e:
            print(f"\n  - Failed to process {file_path}: {e}")

        finally:
            # 5. Cleanup
            if uploaded_file_resource:
                try:
                    client.files.delete(name=uploaded_file_resource.name)
                    print(f"  - Cleanup: Deleted temporary file {uploaded_file_resource.name}")
                except Exception as cleanup_e:
                    print(f"  - Cleanup Warning: Could not delete {uploaded_file_resource.name}: {cleanup_e}")

    print("\nBatch upload processing complete.")

if __name__ == "__main__":
    upload_chunks()
