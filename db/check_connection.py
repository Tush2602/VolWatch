from db.client import Database


def main() -> None:
    db = Database()
    db.run_script("db/schema.sql")
    print("schema applied, connection works")


if __name__ == "__main__":
    main()