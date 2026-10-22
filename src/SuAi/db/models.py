from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column


class Base(MappedAsDataclass, DeclarativeBase):
    pass


class TrainingQaData(Base):
    __tablename__ = "training_qa_data"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    question: Mapped[str]
    answer: Mapped[str]
    good: Mapped[int]
