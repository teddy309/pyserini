##Tutorial by LSS-edited

##follow pysernini-tutorial (colab)
#   - msmarco-passage-demo 'https://colab.research.google.com/github/castorini/anserini-notebooks/blob/master/pyserini_msmarco_passage_demo.ipynb#scrollTo=03sPnM3wWBfJ'
#   - robust04 demo 'https://colab.research.google.com/drive/1GOCyWyYW_fwJMKy5FVWdYHEC2IlvpqZN#scrollTo=KVXWA6WS0aqJ'

###implement experiment results by using pretrained weight of SPARTA, DPR 
import os, sys

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

anserini_root = '.'
sys.path += [os.path.join(anserini_root, 'src/main/python')]



#from pyserini.search import pysearch
#from pyserini.collection import pycollection
#from pyserini.index import pygenerator

topk = 10
selectSearcher = 'SparseSearcher' # ['SimpleSearcher', 'SparseSearcher', 'FaissSearcher']
#   - SimpleSearcher : 
#   - Dense Retrieval(Faiss) : 
#   - Sparse Search(Lucene) : Index 활용
from pyserini.search import SimpleSearcher, get_topics #SimpleSearcher
import json
from pyserini.search.lucene import LuceneSearcher   #Sparse Search
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder #Dense Retrieval

def Retrieval_topk(selectedSearcher, query, n_k):
    if selectedSearcher == 'SimpleSearcher': #비추천. 대신 Lucene 사용하기. 내부적으로 LuceneSearcher 호출함.
        topics = get_topics('msmarco-passage-dev-subset')
        print(f'get_topics: {len(topics)} queries total') #6980 queries total

        #query = topics[1102400]['title'] 
        print('topic-title exmample: ',query) # why do bears hibernate

        searcher = SimpleSearcher.from_prebuilt_index('msmarco-passage')

        hits = searcher.search(query)
        # Prints the first 10 hits
        for i in range(0, n_k):
            jsondoc = json.loads(hits[i].raw)
            print(f'{i+1:2} {hits[i].score:.5f} {jsondoc["contents"][:80]}...')
        ''''
        SimpleSearcher example Results: 
        1 11.00830 Cookbook: Lobster roll Media: Lobster roll A lobster-salad style roll from The L...
        2 10.94310 Calories, Fat, Protein, Fiber, & Carbs In Papa Ginos Sub Lobster Roll. Calories ...
        3 10.81740 Whether you're steaming lobsters for a Maine-style clambake, a classic New Engla...
        4 10.59820 A Lobster Roll is a bread roll filled with bite-sized chunks of lobster meat. Lo...
        5 10.48360 Lobster Roll. This is the classic New England lobster roll: a basic hot dog bun ...
        6 10.31190 We got a lb of shrimp, a lb of lobster and a lobster roll. You're given a bib an...
        7 10.12300 Red Lobster announced Monday that it's now serving an all-new lobster roll as pa...
        8 10.05290 Live lobsters are only available whole. Dead lobsters can be purchased in the sh...
        9 9.96350 To steam live lobster: Fill pot so that water comes up sides about two inches. A...
        10 9.92200 bring the water to a rolling boil and put in lobsters one at a time bring water ...
        '''
    elif selectedSearcher == 'SparseSearcher': ## 1) Sparse-Retrieval
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
        hits = searcher.search(query)

        for i in range(0, n_k):
            print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')
        '''
         1 7157707 11.00830
        2 6034357 10.94310
        3 5837606 10.81740
        4 7157715 10.59820
        5 6034350 10.48360
        6 2900045 10.31190
        7 7157713 10.12300
        8 1584344 10.05290
        9 533614  9.96350
        10 6234461 9.92200
        '''
    elif selectedSearcher == 'FaissSearcher': ## 2) Dense-Retrieval
        #searcher - QueryEncoder,DprQueryEncoder

        encoder = TctColBertQueryEncoder('castorini/tct_colbert-msmarco')
        searcher = FaissSearcher.from_prebuilt_index(
            'msmarco-passage-tct_colbert-hnsw',
            encoder
        )
        #query = 'what is a lobster roll'
        hits = searcher.search(query)

        for i in range(0, n_k):
            print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')
        '''
        FaissSearcher Results:
        1 7157710 70.53741
        2 7157715 70.50041
        3 7157707 70.13805
        4 6034350 69.93666
        5 6321969 69.62682
        6 4112862 69.34587
        7 5515474 69.21356
        8 7157708 69.08415
        9 6321974 69.06841
        10 2920399 69.01737
        '''
    else:
        print('No Searcher selected')
    
    return hits, searcher #list, Searcher_Obj


sampleQuery = 'what is a lobster roll'
print(f'query:{sampleQuery} , searcher:{selectSearcher} with top-{topk} results')

hits_k, searcher = Retrieval_topk(selectSearcher, sampleQuery, topk)





#hits = searcher.search(query)
#print(type(hits), len(hits)) #list
#print(hits[0])

#import json
#jsondoc = json.loads(hits[0].raw)
#print(jsondoc) # keys(id, contents)
#print(jsondoc.keys())
