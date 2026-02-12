import os

from dotenv import load_dotenv
from pymongo import MongoClient


def main() -> None:
    load_dotenv()

    mongo_uri = os.getenv("MONGO_URI_LOCAL") or os.getenv("MONGO_URI") or "mongodb://localhost:27017"
    db_name = os.getenv("DATABASE_NAME", "chakra_hospital")

    legacy_name = "paitents"
    target_name = "patients"

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[db_name]

    collections = set(db.list_collection_names())

    if legacy_name not in collections:
        print(f"No migration needed: '{legacy_name}' collection not found in '{db_name}'.")
        return

    legacy_count = db[legacy_name].count_documents({})

    if target_name not in collections:
        db[legacy_name].rename(target_name)
        print(
            f"Migration complete: renamed '{legacy_name}' -> '{target_name}' in '{db_name}'. "
            f"Moved {legacy_count} documents."
        )
        return

    target_count = db[target_name].count_documents({})
    if target_count == 0:
        db[legacy_name].rename(target_name, dropTarget=True)
        print(
            f"Migration complete: replaced empty '{target_name}' with '{legacy_name}'. "
            f"Moved {legacy_count} documents."
        )
        return

    print(
        f"Migration skipped: both '{legacy_name}' ({legacy_count} docs) and "
        f"'{target_name}' ({target_count} docs) contain data. Resolve manually."
    )


if __name__ == "__main__":
    main()
