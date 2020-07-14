import json
import os

with open("info.json") as f:
    database = json.load(f)


# multiline 주석 : shift + alt + A
class Setting():
    # DB Auth info
    DATABASE_HOST: str = database['db']['host']
    DATABASE_USER: str = database['db']['user']
    DATABASE_PWD: str = database['db']['password']
    DATABASE_DB: str = database['db']['db']

    # other settings....

class Elastic():
    # DB Auth info
    HOST: str = database['ela']['host']
    PORT: str = database['ela']['port']

    # other settings....