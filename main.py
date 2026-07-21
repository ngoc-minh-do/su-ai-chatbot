from SuAi.utils.env import load_env

load_env()

from SuAi.db import engine  # noqa: E402
from SuAi.db.operations import init_db_tables  # noqa: E402
from SuAi.pages.home import main  # noqa: E402

if __name__ == "__main__":
    engine.init_database()
    init_db_tables()
    main()
