CREATE TABLE
  IF NOT EXISTS training_qa_data (
    id SERIAL PRIMARY KEY,
    question text,
    answer text,
    good INTEGER
  );