import sys

import psycopg
from dotenv import dotenv_values


def main() -> None:
    url = dotenv_values(".env.api")["DATABASE_URL"]
    with psycopg.connect(url, connect_timeout=30) as conn:
        count = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
        print("read works, forecast rows:", count)
        try:
            conn.execute("DELETE FROM forecasts WHERE ticker = 'NO_SUCH_TICKER'")
        except psycopg.errors.InsufficientPrivilege:
            print("write blocked: the role is read-only")
        else:
            print("WARNING: the role was allowed to write")
            sys.exit(1)


if __name__ == "__main__":
    main()