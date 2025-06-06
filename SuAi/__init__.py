from .db import engine
from .utils import env

env.load_env()
engine.init_database()
