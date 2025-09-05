import utils
import boto3

from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from opensearchpy import OpenSearch
from langchain.docstore.document import Document
from requests_aws4auth import AWS4Auth
from opensearchpy import RequestsHttpConnection
from langchain_aws import BedrockEmbeddings
from botocore.config import Config

import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("chat")

config = utils.load_config()

opensearch_url = config["managed_opensearch_url"] if "managed_opensearch_url" in config else None
if opensearch_url is None:
    raise Exception ("No OpenSearch URL")

projectName = config["projectName"] if "projectName" in config else "langgraph-nova"
region = config["region"] if "region" in config else "us-west-2"

# Get configuration for document retrieval and hybrid search
enableParentDocumentRetrival = "Enable"
enableHybridSearch = "Enable"

index_name = projectName
session = boto3.Session(region_name=region)
credentials = session.get_credentials()

# AWS4Auth settings (for AWS managed OpenSearch)
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    'es',  # OpenSearch service uses 'es'
    session_token=credentials.token
)

os_client = OpenSearch(
    hosts=[{
        'host': opensearch_url.replace("https://", ""), 
        'port': 443
    }],
    http_compress=True,
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    connection_class=RequestsHttpConnection
)

def lexical_search(query, top_k):
    # lexical search (keyword)
    min_match = 0
    
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "text": {
                                "query": query,
                                "minimum_should_match": f'{min_match}%',
                                "operator":  "or",
                            }
                        }
                    },
                ],
                "filter": [
                ]
            }
        }
    }

    response = os_client.search(
        body=query,
        index=index_name
    )
    # print('lexical query result: ', json.dumps(response))
        
    docs = []
    for i, document in enumerate(response['hits']['hits']):
        if i>=top_k: 
            break
                    
        excerpt = document['_source']['text']
        
        name = document['_source']['metadata']['name']
        # print('name: ', name)

        page = ""
        if "page" in document['_source']['metadata']:
            page = document['_source']['metadata']['page']
        
        url = ""
        if "url" in document['_source']['metadata']:
            url = document['_source']['metadata']['url']            
        
        docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'url': url,
                        'page': page,
                        'from': 'lexical'
                    },
                )
            )
    
    for i, doc in enumerate(docs):
        #print('doc: ', doc)
        #print('doc content: ', doc.page_content)
        
        if len(doc.page_content)>=100:
            text = doc.page_content[:100]
        else:
            text = doc.page_content            
        logger.info(f"--> lexical search doc[{i}]: {text}, metadata:{doc.metadata}")   
        
    return docs

def get_parent_content(parent_doc_id):
    response = os_client.get(
        index = index_name, 
        id = parent_doc_id
    )
    
    source = response['_source']                            
    # print('parent_doc: ', source['text'])   
    
    metadata = source['metadata']    
    #print('name: ', metadata['name'])   
    #print('url: ', metadata['url'])   
    #print('doc_level: ', metadata['doc_level']) 
    
    url = ""
    if "url" in metadata:
        url = metadata['url']
    
    return source['text'], metadata['name'], url

def get_embedding():
    LLM_embedding = [
        {
            "bedrock_region": "us-west-2", # Oregon
            "model_type": "titan",
            "model_id": "amazon.titan-embed-text-v2:0"
        },
        {
            "bedrock_region": "us-east-1", # N.Virginia
            "model_type": "titan",
            "model_id": "amazon.titan-embed-text-v2:0"
        },
        {
            "bedrock_region": "us-east-2", # Ohio
            "model_type": "titan",
            "model_id": "amazon.titan-embed-text-v2:0"
        }
    ]
    
    selected_embedding = 0
    embedding_profile = LLM_embedding[selected_embedding]
    bedrock_region = embedding_profile['bedrock_region']
    model_id = embedding_profile['model_id']
    logger.info(f"selected_embedding: {selected_embedding}, bedrock_region: {bedrock_region}, model_id: {model_id}")
    
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region, 
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    
    bedrock_embedding = BedrockEmbeddings(
        client=boto3_bedrock,
        region_name = bedrock_region,
        model_id = model_id
    )  
    
    return bedrock_embedding

def retrieve_documents_from_opensearch(query, top_k):
    logger.info(f"###### retrieve_documents_from_opensearch ######")

    # Vector Search
    bedrock_embedding = get_embedding()       

    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name=index_name,  
        is_aoss = False,
        #engine="faiss",  # default: nmslib
        embedding_function=bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=awsauth,
        connection_class=RequestsHttpConnection
    )  
    
    relevant_docs = []
    if enableParentDocumentRetrival == 'Enable':
        result = vectorstore_opensearch.similarity_search_with_score(
            query = query,
            k = top_k*2,  
            search_type="script_scoring",
            pre_filter={"term": {"metadata.doc_level": "child"}}
        )
        logger.info(f"result: {result}")
                
        relevant_documents = []
        docList = []
        for re in result:
            if 'parent_doc_id' in re[0].metadata:
                parent_doc_id = re[0].metadata['parent_doc_id']
                doc_level = re[0].metadata['doc_level']
                logger.info(f"doc_level: {doc_level}, parent_doc_id: {parent_doc_id}")
                        
                if doc_level == 'child':
                    if parent_doc_id in docList:
                        logger.info(f"duplicated")
                    else:
                        relevant_documents.append(re)
                        docList.append(parent_doc_id)                        
                        if len(relevant_documents)>=top_k:
                            break
                                    
        # print('relevant_documents: ', relevant_documents)    
        for i, doc in enumerate(relevant_documents):
            if len(doc[0].page_content)>=100:
                text = doc[0].page_content[:100]
            else:
                text = doc[0].page_content            
            logger.info(f"--> vector search doc[{i}]: {text}, metadata:{doc[0].metadata}")

        for i, document in enumerate(relevant_documents):
                logger.info(f"## Document(opensearch-vector) {i+1}: {document}")
                
                parent_doc_id = document[0].metadata['parent_doc_id']
                doc_level = document[0].metadata['doc_level']
                #print(f"child: parent_doc_id: {parent_doc_id}, doc_level: {doc_level}")
                
                content, name, url = get_parent_content(parent_doc_id) # use pareant document
                #print(f"parent_doc_id: {parent_doc_id}, doc_level: {doc_level}, url: {url}, content: {content}")
                
                relevant_docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            'name': name,
                            'url': url,
                            'doc_level': doc_level,
                            'from': 'vector'
                        },
                    )
                )
    else: 
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = query,
            k = top_k
        )
        
        for i, document in enumerate(relevant_documents):
            logger.info(f"## Document(opensearch-vector) {i+1}: {document}")   
            name = document[0].metadata['name']
            url = document[0].metadata['url']
            content = document[0].page_content
                   
            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'name': name,
                        'url': url,
                        'from': 'vector'
                    },
                )
            )
    # print('the number of docs (vector search): ', len(relevant_docs))

    # Lexical Search
    if enableHybridSearch == 'Enable':
        relevant_docs += lexical_search(query, top_k)    

    return relevant_docs

