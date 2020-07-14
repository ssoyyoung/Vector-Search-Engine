import time
import pymysql


class imgtovec:
    base_img_path = "/mnt/piclick/piclick.tmp/AIMG/"

    # [0'id',1'au_id',2'cat_key',3'i_key',4'i_category','5img_url','6click_url','7save_path','8save_name','9group_id','10status','11cre_tt','12appr_tt','13rej_tt','14soldout','15gs_bucket','16product_set','17product_id','18label','19hash','20download']
    # ['id','au_id','p_key','p_category','img_url','click_url','save_path','save_name','status','imp_cnt','click_cnt','cre_tt','appr_tt','rej_tt']
    def connect_db(self, sql):
        print('Connecting to db......')
        sql_time = time.time()
        conn = pymysql.connect(host='piclick-ads.mysql.rds.aliyuncs.com',
                               user='piclick',
                               password='psr0035!',
                               db='pirs_test')

        curs = conn.cursor()
        curs.execute(sql)
        data = curs.fetchall()
        curs.close()
        conn.close()

        print('Interface time for querying sql', time.time() - sql_time)
        return data

    def vector_extract(self, service=False, data='/home/piclick/vector_api/test_img/', batch_size=100):

        yolov3 = Yolov3()

        if service:
            res = yolov3.vector_extraction_batch(data=data, batch_size=batch_size)
        else:
            res = yolov3.vector_extraction_service(data)

        return res