import json
import time
from elasticsearch import Elasticsearch, helpers
from .utils import *


class Elk:
    es = Elasticsearch('172.16.0.240:9200', timeout=30, max_retries=10, retry_on_timeout=True)

    search_index = 'pirs_'
    save_index = 'pirs_'

    def create_index(self, index):
        with open('mapping.json', 'r') as f:
            mapping = json.load(f)

        if type(index) == list:
            for idx in index:
                if not self.es.indices.exists(index=idx):
                    self.es.indices.create(index=idx, body=mapping)
        else:
            if not self.es.indices.exists(index=index):
                self.es.indices.create(index=index, body=mapping)

    def bulk_api_res_yolo(self, total_vec, data_dict):
        docs = []
        es_time = time.time()
        for v_idx in total_vec.keys():
            if 'U' in v_idx:
                new_v_idx = v_idx.replace('U', 'WM')
                new_v_idx = new_v_idx.lower()
            else:
                new_v_idx = v_idx.lower()

            if len(new_v_idx) == 4:
                new_v_idx = new_v_idx[0] + '_' + new_v_idx[1] + '_' + new_v_idx[2:]
            elif len(new_v_idx) == 5:
                new_v_idx = new_v_idx[0:2] + '_' + new_v_idx[2] + '_' + new_v_idx[3:]

            new_index = self.save_index + new_v_idx
            self.create_index(self, new_index)
            for count in range(len(total_vec[v_idx])):
                docs.append({
                    '_index': new_index,
                    '_source': {
                        "id": data_dict[total_vec[v_idx][count]['img_path']][0],
                        "au_id": data_dict[total_vec[v_idx][count]['img_path']][1],
                        "cat_key": data_dict[total_vec[v_idx][count]['img_path']][2],
                        "i_key": data_dict[total_vec[v_idx][count]['img_path']][3],
                        "img_url": data_dict[total_vec[v_idx][count]['img_path']][4],
                        "click_url": data_dict[total_vec[v_idx][count]['img_path']][5],
                        "img_path": data_dict[total_vec[v_idx][count]['img_path']][6],
                        "group_id": data_dict[total_vec[v_idx][count]['img_path']][7],
                        "status": data_dict[total_vec[v_idx][count]['img_path']][8],
                        "p_key": data_dict[total_vec[v_idx][count]['img_path']][3],  # 임시
                        "gs_bucket" : "https://storage.cloud.google.com"+data_dict[total_vec[v_idx][count]['img_path']][9][4:],
                        "raw_box": np.array(total_vec[v_idx][count]['raw_box']).tolist(),
                        "yolo_vector_mac": encode_array(total_vec[v_idx][count]['yolo_vector_mac']),
                        "yolo_vector_spoc": encode_array(total_vec[v_idx][count]['yolo_vector_spoc']),
                        "yolo_vector_spoc_w/o_norm": encode_array(total_vec[v_idx][count]['yolo_vector_spoc']),
                        "yolo_vector_cat": encode_array(total_vec[v_idx][count]['yolo_vector_cat']),
                        "resnet_vector": encode_array(total_vec[v_idx][count]['resnet_vector']),
                        "box_statue": total_vec[v_idx][count]['state'],
                        "@timestamp": utc_time()
                    }
                })
        helpers.bulk(self.es, docs)
        print('Interface time for sending to elastic', time.time() - es_time)

    def search_vec(self, search_index, search_vec, size=1):
        st_time = time.time()
        #search_index = self.search_index + search_index.lower()
        res = self.es.search(
            index=search_index,
            body={
                "_source": {
                    "includes": ["_index", "img_url"]
                },
                "query": {"function_score": {
                    "boost_mode": "replace",
                    "script_score": {"script": {
                        "source": "binary_vector_score", "lang": "knn",
                        "params": {
                            "cosine": True,
                            "field": "vector",
                            "encoded_vector": search_vec
                        }
                    }
                    }
                }
                },
                "size": size
            },
            request_timeout=300
        )
        print('Interface time for searching vector', time.time() - st_time)
        return json.dumps(res, ensure_ascii=True, indent='\t')

    def multi_search_vec(self, idx_list, vb_list, k, vector_type):
        st_time = time.time()
        total_body=''
        for num in range(len(idx_list)):
            index = '{"index":"'+str(idx_list[num])+'"}'
            body_data = '{"_source": {"includes": ["_index", "_score", "box_statue", "cat_key", "gs_bucket", "click_url", "image_path", "raw_box"]},"query": {"function_score": {"boost_mode": "replace","script_score": {"script": {"source": "binary_vector_score", "lang": "knn","params": {"cosine": true,"field": "'+vector_type+'","encoded_vector": "' + vb_list[num] + '"}}}}},"size": "'+str(k)+'"}'
            total_body = total_body+index+'\n'+body_data+'\n'

        res = self.es.msearch(
            body=total_body
        )
        print('Interface time for searching vector', time.time() - st_time)
        return json.dumps(res, ensure_ascii=True, indent='\t')

    def save_vector_ps(self, vec, c_key, img_url):
        now = datetime.now()
        index = "search_img"
        doc = {
            "vector" : encode_array(vec),
            "c_key" : c_key,
            "img_url" : img_url,
            "ymd" : str(now.year)+str(now.month)+str(now.day),
            "hour" : str(now.hour)
        }

        self.es.index(index=index, doc_type="_doc", body=doc)


