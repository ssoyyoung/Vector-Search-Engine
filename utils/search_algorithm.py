import json

with open('utils/category_mapping_search.json', 'r') as f:
    mapping = json.load(f)


def change_index(ori_idx, sex):
    idx_code = mapping[ori_idx]
    idx = idx_code[0].lower()+'_'+idx_code[1:]

    if sex == 'W':
        search_idx = '*w*'+idx
    elif sex == 'M':
        search_idx = '*_'+idx
    elif sex == 'N' or 'A':
        search_idx = '*'+'_'+idx
    else:
        return -1

    return search_idx


