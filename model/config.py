import json
import os

with open("info.json") as f:
    database = json.load(f)


class Setting():
    DATABASE_HOST: str = database['db']['host']
    DATABASE_USER: str = database['db']['user']
    DATABASE_PWD: str = database['db']['password']
    DATABASE_DB: str = database['db']['db']