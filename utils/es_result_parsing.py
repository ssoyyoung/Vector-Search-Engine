def es_parsing(res, rb_list=[], cate_list=[], multi=True):
    es_total = []
    res = eval(res.replace('false', 'False'))

    if multi: #boxes in one img
        for idx in range(len(res['responses'])):
            es = []
            es_obj = {}
            for count in range(len(res['responses'][idx]['hits']['hits'])):
                data = res['responses'][idx]['hits']['hits'][count]['_source']
                data['box_state'] = data.pop('box_statue')
                data['_score'] = res['responses'][idx]['hits']['hits'][count]['_score']
                es.append(data)
            es_obj['input_raw_box'] = rb_list[idx]
            es_obj['cat_key'] = cate_list[idx]
            es_obj['search_result'] = es
            es_total.append(es_obj)
    else: #one img
        es = []
        es_obj = {}
        for count in range(len(res['hits']['hits'])):
            data = res['hits']['hits'][count]['_source']
            data['_score']=res['hits']['hits'][count]['_score']
            data['raw_box']=None
            es.append(data)
        es_obj['input_raw_box'] = None
        es_obj['cat_key'] = None
        es_obj['search_result'] = es
        es_total.append(es_obj)

    return es_total