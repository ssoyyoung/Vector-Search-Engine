import time
import pymysql
from model.config import Setting

class SQL:
    base_img_path = "/mnt/piclick/piclick.tmp/AIMG/"

    def connect_db(self, sql_query):
        sql_time = time.time()

        print('Connecting to db......')
        conn = pymysql.connect( 
                                host=Setting.DATABASE_HOST,
                                user=Setting.DATABASE_USER,
                                password=Setting.DATABASE_PWD,
                                db=Setting.DATABASE_DB
                                )



        curs = conn.cursor()
        curs.execute(sql_query)
        data = curs.fetchall()
        curs.close()
        conn.close()

        print('Interface time for querying sql', time.time() - sql_time)

        return data
