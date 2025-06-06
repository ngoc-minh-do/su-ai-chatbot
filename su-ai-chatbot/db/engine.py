import logging as loggingLib

from sqlalchemy import create_engine, text

from utils import constants, logging

logger = logging.get_logger(__name__)


def init_database():
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


init_database()

engine = create_engine(
    constants.db_connection_string, echo=logger.level == loggingLib.DEBUG
)
