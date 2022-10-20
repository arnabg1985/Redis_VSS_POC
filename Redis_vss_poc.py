import random
import numpy as np
import pandas as pd
import time
import redis
from redis.commands.search.field import VectorField
from redis.commands.search.field import TextField
from redis.commands.search.field import TagField
from redis.commands.search.query import Query
from redis.commands.search.result import Result
import requests
import json

datafile_path = "C:\\Users\\mainaksaha\\fine_food_reviews_with_embeddings_1k.csv"  # for your convenience, we precomputed the embeddings
df_food_review = pd.read_csv(datafile_path)
#df_food_review.head()
df_food_review = df_food_review[['ProductId', 'Text']]
df_food_review.head()


open_ai_embedding_endpoint = 'https://api.openai.com/v1/embeddings'
auth_token = 'Bearer <API_KEY>'

#get embedding for a single string
def get_text_embedding(text): 
    try:       
        header_values = {
        'Authorization': auth_token
        }
        payload = {
        "model": "text-similarity-babbage-001",
        "input": text
        }
        response = requests.post(open_ai_embedding_endpoint, headers = header_values, json = payload)
        response_json =  response.json()
        embedding = response_json['data'][0]['embedding']
        #print(embedding)
        return embedding
    except:
        return None

#get embedding for a dataframe
def get_text_embeddings(df_sentenses):
    list_of_embedding_vectors = []

    for i in range(df_sentenses.shape[0]):
        embedding_vector = get_text_embedding(df_sentenses.loc[i]['Text'])
        list_of_embedding_vectors.append(embedding_vector)

    #print(list_of_embedding_vectors)
    df_sentenses['embedding_vector'] = list_of_embedding_vectors
    return df_sentenses
    

df_embeddings = get_text_embeddings(df_food_review) 
df_embeddings.head()

print(len(df_embeddings.loc[0]['embedding_vector']))
print(len(df_embeddings.loc[107]['embedding_vector']))

redis_conn = redis.Redis( host = '10.1.255.226', port = '6379')

redis_conn.flushall()



##create HNSW index
def create_hnsw_index(distance_metric, run_time, vector_dimension):
    redis_command = ["FT.CREATE", "idx"]
    redis_command +=["SCHEMA", "product_id", "TEXT", "vector_field", "VECTOR", "HNSW", "12",
                        "TYPE", "FLOAT32",
                        "DIM", str(vector_dimension),
                        "DISTANCE_METRIC", str(distance_metric),
                        "INITIAL_CAP", 300,
                        "M", 40,
                        "EF_CONSTRUCTION", 200]
    print(redis_command)
    redis_conn.execute_command(*redis_command)        
        
create_hnsw_index('COSINE', 20, 2048)



#load articles into redis hash
import numpy as np
def load_vectors(client, df_embeddings):
    #pipeline the 300 articles in one go
    p = client.pipeline(transaction=False)
    for index, row in df_embeddings.iterrows():    
        #hash key
        key= row['ProductId']
        #hash fields
        #summary=row['Summary']
        embedding_vector_np_array = np.array(row['embedding_vector'], dtype=np.float32)
        summery_vector = embedding_vector_np_array.astype(np.float32).tobytes()
        #summary_data_mapping ={'summary':summary, 'vector_field':summery_vector}
        summary_data_mapping ={'product_id': key, 'vector_field':summery_vector}
        
        p.hset(key,mapping=summary_data_mapping)
    p.execute()
            
 
 load_vectors(redis_conn, df_embeddings )
 
 
from redis.commands.search import Search
from redis.commands.search.query import Query



user_query='recession in european markets'
#user_query='wealth management'
#user_query = 'vegeterian food'
#e = model.encode(user_query)
e = get_text_embedding(user_query)
e_bytes = np.array(e).astype(np.float32).tobytes()

q = Query(f'*=>[KNN $K @vector_field $BLOB]').return_fields('product_id').sort_by('__vector_field_score').paging(0,10).dialect(2)

#parameters to be passed into search
params_dict = {"K": 10, "BLOB": e_bytes}

docs = redis_conn.ft().search(q,params_dict).docs
list_of_ids = []

for doc in docs:    
    list_of_ids.append(doc.product_id)
    
print(list_of_ids)

df_embeddings_filtered = df_embeddings.query('ProductId in @list_of_ids')
print(df_embeddings_filtered['Text'])
df_embeddings_filtered.head()
