import os
import shutil
import sys

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Target files and directories to be deleted
    targets = [
        os.path.join(root, "llama_index_storage"),           # Vector database
        os.path.join(root, "cache", "processed_messages.json"), # Ingestion cache
        os.path.join(root, "discord_data.db"),               # Raw messages SQLite DB
    ]
    
    print("⚠️  WARNING: You are about to permanently delete the Vector DB and all associated ingestion caches.")
    print("This will force the bot to completely re-ingest all chat history.")
    print("\nTargets to delete:")
    for t in targets:
        print(f" - {t}")
        
    confirm = input("\nType 'YES' to proceed with deletion: ")
    if confirm != "YES":
        print("Aborted.")
        sys.exit(0)
        
    print("\nDeleting...")
    for target in targets:
        if os.path.exists(target):
            try:
                if os.path.isdir(target):
                    shutil.rmtree(target)
                    print(f"✅ Deleted directory: {target}")
                else:
                    os.remove(target)
                    print(f"✅ Deleted file: {target}")
            except Exception as e:
                print(f"❌ Failed to delete {target}: {e}")
        else:
            print(f"⏭️  Skipped (not found): {target}")
            
    print("\n🎉 Reset complete! The next time you run main.py, it will build the ingestion database from scratch.")

if __name__ == "__main__":
    main()
