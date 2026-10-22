from sqlalchemy.orm import Session

from ..db.engine import get_engine
from ..db.models import TrainingQaData
from ..utils import logging

logger = logging.get_logger(__name__)


def init_db_tables():
    from ..db.utils import create_database

    create_database()


def create_training_qa_data(question: str, answer: str, good: int) -> None:
    try:
        with Session(get_engine()) as session:
            data = TrainingQaData(question=question, answer=answer, good=good)
            session.add(data)
            session.commit()
            logger.info(f"create_training_qa_data: {data}")
    except Exception as e:
        logger.error(f"Failed to save training QA data: {e}")
        raise RuntimeError(f"Failed to save training data: {e}") from e
