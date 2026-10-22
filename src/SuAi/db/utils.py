from ..db.engine import get_engine
from ..db.models import Base
from ..utils import logging

logger = logging.get_logger(__name__)


def create_database() -> None:
    logger.info("Creating tables...")
    Base.metadata.create_all(get_engine())


def drop_database(force=False) -> None:
    if not force:
        result = input(
            "Are you sure you want to drop all tables? This action cannot be undone. Type 'yes' to confirm: "
        )
        if result.lower() != "yes":
            logger.info("Database drop cancelled.")
            return

    logger.info("Dropping tables...")
    Base.metadata.drop_all(get_engine())


def reset_database() -> None:
    result = input(
        "Are you sure you want to reset the database? This action will drop all tables and recreate them. Type 'yes' to confirm: "
    )
    if result.lower() != "yes":
        logger.info("Database reset cancelled.")
        return

    drop_database(force=True)
    create_database()
