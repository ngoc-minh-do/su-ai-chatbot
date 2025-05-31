from sqlalchemy import create_engine, text

from utils import constants, logging

logger = logging.get_logger(__name__)


def main():
    logger.info("Initializing database...")

    base_url = "/".join(constants.db_connection_string.split("/")[:-1]) + "/postgres"

    engine = create_engine(base_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        dbname = constants.db_connection_string.split("/")[-1]
        cursor = conn.execute(
            text(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}'")
        )

        if not cursor.fetchone():
            logger.info(f"Database '{dbname}' does not exist, creating it...")
            conn.execute(text(f"CREATE DATABASE {dbname}"))

    engine = create_engine(constants.db_connection_string, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn, open("schema.sql", "r") as schema_file:
        sql = text(schema_file.read())
        conn.execute(sql)
        logger.info("Database initialized and table created.")


if __name__ == "__main__":
    main()
