import faiss
from sklearn.preprocessing import normalize

from config import ExtractorCofing, ResultConfig
from vectorAPI.extVector import extractVec

# default settings 
baseDir = ResultConfig.BASE_DIR
fvecsBin = ResultConfig.FVECS_BIN
fnamesTxt = ResultConfig.FNAMES
indexType = ResultConfig.INDEX_TYPE

def getVec(fname):
    res = extractVec(fname)

    for key in res.keys():
        print(res[key]['res'].shape)

        # TODO : vector extrator 추가
    

def LoadIndex(cate, vec, k=10):
    dim = ExtractorCofing.DIM
    fvec_file = baseDir + cate + fvecsBin
    index_file = f'{fvec_file}.{indexType}.index'

    index = faiss.read_index(index_file)

    if indexType == 'hnsw': index.hnsw.efSearch = 256

    _, idxs = index.search(normalize(vec), k)

    print(idxs)


if __name__ == "__main__":
    getVec("test.jpg")