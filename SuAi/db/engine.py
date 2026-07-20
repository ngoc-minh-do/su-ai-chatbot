import logging as loggingLib
import re

from sqlalchemy import create_engine, text

from ..utils import constants, logging

logger = logging.get_logger(__name__)

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

def _validate_identifier(name: str) -> str:
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid database identifier: {name}")
    return name


def init_database():
    logger.info("Initializing database...")

    base_url = "/".join(constants.db_connection_string.split("/")[:-1]) + "/postgres"

    engine = create_engine(base_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        dbname = _validate_identifier(constants.db_connection_string.split("/")[-1])
        cursor = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
            {"dbname": dbname},
        )

        if not cursor.fetchone():
            logger.info(f"Database '{dbname}' does not exist, creating it...")
            conn.execute(text(f"CREATE DATABASE {dbname}"))


engine = create_engine(
    constants.db_connection_string, echo=logger.level == loggingLib.DEBUG
)
