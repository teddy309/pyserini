#!/bin/bash

## Code by LSS(github: teddy309)
## date: 2022.07.14 ~ Current(on updating)

#. conda/bin/activate
#conda activate pyserini #python=3.8+

<< "END_conda_settings"
conda install -c pytorch faiss-gpu
conda install pytorch
END_conda_settings

##follow pysernini-tutorial (colab)
#   - robust04 demo 'https://colab.research.google.com/drive/1GOCyWyYW_fwJMKy5FVWdYHEC2IlvpqZN#scrollTo=KVXWA6WS0aqJ'
#   - msmarco-passage-demo 'https://colab.research.google.com/github/castorini/anserini-notebooks/blob/master/pyserini_msmarco_passage_demo.ipynb#scrollTo=03sPnM3wWBfJ'
<< "END_AnseriniSetting"
apt-get update
apt-get install -y openjdk-11-jdk-headless -qq 
apt-get install maven -qq

pip install Cython
pip install pyjnius

#install anserini.git
mvn clean package appassembler:assemble -DskipTests -Dmaven.javadoc.skip=true
cd eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make

echo "target for '.jar' files"
ls target

wget https://www.dropbox.com/s/mdoly9sjdalh44x/lucene-index.robust04.pos%2Bdocvectors%2Brawdocs.tar.gz
tar xvfz lucene-index.robust04.pos+docvectors+rawdocs.tar.gz
END_AnseriniSetting


## 
##
export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
python lss_ds_tutorial.py

#python -m pyserini.eval.trec_eval -c -mrecall.1000 -mmap msmarco-passage-dev-subset run-msmarco-passage-bm25.txt
#run-msmarco-passage-bm25.txt
#
#

#brute-force
#python -m pyserini.search.faiss \
#  --index wikipedia-dpr-multi-bf \
#  --topics dpr-nq-test \
#  --encoded-queries dpr_multi-nq-test \
#  --output runs/run.dpr.nq-test.multi.bf.trec \
#  --batch-size 36 --threads 12