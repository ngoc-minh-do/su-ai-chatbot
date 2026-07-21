from SuAi.db import engine
from SuAi.db.operations import init_db_tables
from SuAi.pages.home import main

if __name__ == "__main__":
    engine.init_database()
    init_db_tables()
    main()
