
from cdqa.pipeline.cdqa_sklearn import QAPipeline
from cdqa.utils.filters import filter_paragraphs
import os


cur_path = os.path.dirname(os.path.abspath(__file__))
reader_path = os.path.join(cur_path,'models/bert_qa.joblib')


def fetch_cdqa_pipeline(reader_path):
    cdqa_pipeline = QAPipeline(reader=reader_path)
    return cdqa_pipeline

def build_knowledge_base(ip_path):
    df = pd.read_csv(ip_path, converters={'paragraphs': literal_eval})
    df = filter_paragraphs(df)
    return df


def predict_answer(query,cdqa_pipeline,num_answers):
    results=[]
    if not query.endswith('?'):
        query = query + '?'
    # Sending a question to the pipeline and getting prediction
    predictions = cdqa_pipeline.predict(query=query,n_predictions=num_answers)
    for i,prediction in enumerate(predictions):
        prediction = list(prediction)
        result = {'Rank':(i+1),'answer': prediction[0],'title': prediction[1],'paragraph':prediction[2]}
        results.append(result)
    print(results)
    return results

def get_answer(df,cdqa_pipeline, query,num_answers):
    # Fitting the retriever to the list of documents in the dataframe
    cdqa_pipeline.fit_retriever(df=df)
    results = predict_answer(query,cdqa_pipeline,num_answers)
    return results

if __name__=='__main__':
    cdqa_pipeline = fetch_cdqa_pipeline(reader_path)
    df = build_knowledge_base('../../data')
    #get_answer(df,cdqa_pipeline,'who has been affected by the coronovirus in indian army',3)
    get_answer(df,cdqa_pipeline,'what do we analyse to carry out our mission To carry out our mission?',3)




'''
import os
import pandas as pd
from ast import literal_eval

from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline


from cdqa.utils.download import download_model, download_bnpp_data, download_squad

download_bnpp_data(dir='./data/bnpp_newsroom_v1.1/')
download_model(model='bert-squad_1.1', dir='./models')
download_squad(dir='./data/squad_v1.1/')


df = pd.read_csv('./data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv', converters={'paragraphs': literal_eval})
df = filter_paragraphs(df)
cdqa_pipeline = QAPipeline(reader='models/bert_qa.joblib') # use 'distilbert_qa.joblib' for DistilBERT instead of BERT
cdqa_pipeline.fit_retriever(df=df)

query = 'Since when does the Excellence Program of BNP Paribas exist?'
prediction = cdqa_pipeline.predict(query)
'''

from cdqa.utils.converters import pdf_converter
import pandas as pd
from ast import literal_eval

df = pd.read_csv('./data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv', converters={'paragraphs': literal_eval})
df = df[['title','paragraphs']]
df = filter_paragraphs(df)
