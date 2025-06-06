from sqlalchemy.orm import Session

from ..db.engine import engine
from ..db.models import TrainingQaData
from ..db.utils import create_database
from ..utils import logging

logger = logging.get_logger(__name__)

create_database()


def create_training_qa_data(question: str, answer: str, good: int) -> None:
    with Session(engine) as session:
        data = TrainingQaData(question=question, answer=answer, good=good)
        session.add(data)
        session.commit()
        logger.info(f"create_training_qa_data: {data}")
