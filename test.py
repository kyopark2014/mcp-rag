import application.utils as utils
import boto3
import traceback
import fitz
import os
import time
import json
import re
from PIL import Image

from opensearchpy import OpenSearch
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain_community.embeddings import BedrockEmbeddings
from botocore.config import Config
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.document import Document
from langchain_aws import ChatBedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib import parse
from pypdf import PdfReader   
from io import BytesIO

config = utils.load_config()

opensearch_url = config['managed_opensearch_url']
projectName = config['projectName']
s3_bucket = config['s3_bucket']
s3_arn = config['s3_arn']
sharing_url = config['sharing_url']
path = sharing_url + '/'
ocr = "Disable"
s3_prefix = "docs"
doc_prefix = s3_prefix+'/'
meta_prefix = "metadata/"
contextual_embedding = 'Enable'
multi_region = 'Disable'
enableParentDocumentRetrival = 'true'
model_name = 'Claude 3.5 Sonnet'

# AWS IAM authentication settings
region = config['region']
service = 'es'  # Managed OpenSearch
maxOutputTokens = 4096
HUMAN_PROMPT = "\n\nHuman:"

s3_client = boto3.client('s3')  


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

# Titan Embedding v2 settings (dimension = 1024)
titan_embedding_v2 = [
    {
        "bedrock_region": "us-west-2",  # Oregon
        "model_type": "titan",
        "model_id": "amazon.titan-embed-text-v2:0"
    },
    {
        "bedrock_region": "us-east-1",  # N.Virginia
        "model_type": "titan",
        "model_id": "amazon.titan-embed-text-v2:0"
    },
    {
        "bedrock_region": "us-east-2",  # Ohio
        "model_type": "titan",
        "model_id": "amazon.titan-embed-text-v2:0"
    }
]

LLM_embedding = titan_embedding_v2  # titan_embedding_v2_single

selected_embedding = 0
multi_region = "Disable"  # or "Enable"

def get_embedding():
    global selected_embedding
    profile = LLM_embedding[selected_embedding]
    bedrock_region = profile['bedrock_region']
    model_id = profile['model_id']
    print(f'selected_embedding: {selected_embedding}, bedrock_region: {bedrock_region}')
    
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region, 
        config=Config(
            retries={
                'max_attempts': 30
            }
        )
    )
    
    bedrock_embedding = BedrockEmbeddings(
        client=boto3_bedrock,
        region_name=bedrock_region,
        model_id=model_id
    )  
    
    if multi_region == "Enable":
        selected_embedding = selected_embedding + 1
        if selected_embedding == len(LLM_embedding):
            selected_embedding = 0
    else:
        selected_embedding = 0
    
    return bedrock_embedding

bedrock_embeddings = get_embedding()

index_name = projectName
vectorstore = OpenSearchVectorSearch(
    index_name=index_name,  
    is_aoss = False,
    #engine="faiss",  # default: nmslib
    embedding_function=bedrock_embeddings,
    opensearch_url=opensearch_url,
    http_auth=awsauth,
    connection_class=RequestsHttpConnection
)  

def get_model_info(model):
    nova_pro_models = [   # Nova Pro
        {   
            "bedrock_region": "us-west-2", # Oregon
            "model_type": "nova",
            "model_id": "us.amazon.nova-pro-v1:0"
        },
        {
            "bedrock_region": "us-east-1", # N.Virginia
            "model_type": "nova",
            "model_id": "us.amazon.nova-pro-v1:0"
        },
        {
            "bedrock_region": "us-east-2", # Ohio
            "model_type": "nova",
            "model_id": "us.amazon.nova-pro-v1:0"
        }
    ]

    nova_lite_models = [   # Nova Lite
        {   
            "bedrock_region": "us-west-2", # Oregon
            "model_type": "nova",
            "model_id": "us.amazon.nova-pro-v1:0"
        },
        {
            "bedrock_region": "us-east-1", # N.Virginia
            "model_type": "nova",
            "model_id": "us.amazon.nova-pro-v1:0"
        },
        {
            "bedrock_region": "us-east-2", # Ohio
            "model_type": "nova",
            "model_id": "us.amazon.nova-pro-v1:0"
        }
    ]

    claude_sonnet_3_5_v1_models = [   # Sonnet 3.5 V1
        {
            "bedrock_region": "us-west-2", # Oregon
            "model_type": "claude",
            "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
        },
        {
            "bedrock_region": "us-east-1", # N.Virginia
            "model_type": "claude",
            "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
        },
        {
            "bedrock_region": "us-east-2", # Ohio
            "model_type": "claude",
            "model_id": "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        }
    ]

    claude_sonnet_3_5_v2_models = [   # Sonnet 3.5 V2
        {
            "bedrock_region": "us-west-2", # Oregon
            "model_type": "claude",
            "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0"
        },
        {
            "bedrock_region": "us-east-1", # N.Virginia
            "model_type": "claude",
            "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        },
        {
            "bedrock_region": "us-east-2", # Ohio
            "model_type": "claude",
            "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        }
    ]

    claude_sonnet_3_0_models = [   # Sonnet 3.0
        {
            "bedrock_region": "us-west-2", # Oregon
            "model_type": "claude",
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
        },
        {
            "bedrock_region": "us-east-1", # N.Virginia
            "model_type": "claude",
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0"
        }
    ]

    claude_haiku_3_5_models = [   # Haiku 3.5 
        {
            "bedrock_region": "us-west-2", # Oregon
            "model_type": "claude",
            "model_id": "anthropic.claude-3-5-haiku-20241022-v1:0"
        },
        {
            "bedrock_region": "us-east-1", # N.Virginia
            "model_type": "claude",
            "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        },
        {
            "bedrock_region": "us-east-2", # Ohio
            "model_type": "claude",
            "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        }
    ]

    claude_3_7_sonnet_models = [   # Sonnet 3.7
        {
            "bedrock_region": "us-west-2", # Oregon
            "model_type": "claude",
            "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        },
        {
            "bedrock_region": "us-east-1", # N.Virginia
            "model_type": "claude",
            "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        },
        {
            "bedrock_region": "us-east-2", # N.Ohio
            "model_type": "claude",
            "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        }
    ]

    claude_models = [
        {   # Claude 3.7 Sonnet
            "bedrock_region": "us-west-2", # Oregon   
            "model_type": "claude",
            "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        },
        {
            "bedrock_region": "us-east-1", # N.Virginia
            "model_type": "claude",
            "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        },
        {   # Claude 3.5 Sonnet v1
            "bedrock_region": "us-west-2", # Oregon
            "model_type": "claude",
            "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
        },
        {
            "bedrock_region": "us-east-1", # N.Virginia
            "model_type": "claude",
            "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0"
        },
        {
            "bedrock_region": "us-east-2", # Ohio
            "model_type": "claude",
            "model_id": "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        },
        {   # Claude 3.5 Sonnet v2
            "bedrock_region": "us-west-2", # Oregon
            "model_type": "claude",
            "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0"
        },
        {
            "bedrock_region": "us-east-1", # N.Virginia
            "model_type": "claude",
            "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        },
        {
            "bedrock_region": "us-east-2", # Ohio
            "model_type": "claude",
            "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        }
    ]

    if model == 'Nova Pro':
        return nova_pro_models
    elif model == 'Nova Lite':
        return nova_lite_models
    elif model == 'Claude 3.7 Sonnet':
        return claude_3_7_sonnet_models 
    elif model == 'Claude 3.5 Sonnet':
        return claude_sonnet_3_5_v2_models  # claude_sonnet_3_5_v1_models
    elif model == 'Claude 3.0 Sonnet':
        return claude_sonnet_3_0_models    
    elif model == 'Claude 3.5 Haiku':
        return claude_models
    else:
        return claude_models

def get_model():
    LLM_for_chat = get_model_info(model_name)
    
    profile = LLM_for_chat[0]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    print(f'bedrock_region: {bedrock_region}, modelId: {modelId}')
                              
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
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    llm = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )    
            
    return llm

def get_contextual_docs_from_chunks(whole_doc, splitted_docs): # per chunk
    contextual_template = (
        "<document>"
        "{WHOLE_DOCUMENT}"
        "</document>"
        "Here is the chunk we want to situate within the whole document."
        "<chunk>"
        "{CHUNK_CONTENT}"
        "</chunk>"
        "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk."
        "Answer only with the succinct context and nothing else."
        "Put it in <result> tags."
    )          
    
    contextual_prompt = ChatPromptTemplate([
        ('human', contextual_template)
    ])

    contexualized_docs = []
    contexualized_chunks = []
    for i, doc in enumerate(splitted_docs):
        # chat = get_contexual_retrieval_chat()
        llm = get_model()
        
        contexual_chain = contextual_prompt | llm
            
        response = contexual_chain.invoke(
            {
                "WHOLE_DOCUMENT": whole_doc.page_content,
                "CHUNK_CONTENT": doc.page_content
            }
        )
        # print('--> contexual chunk: ', response)
        output = response.content
        contextualized_chunk = output[output.find('<result>')+8:output.find('</result>')]
        print('contextualized_chunk: ', contextualized_chunk)
        contextualized_chunk.replace('\n', '')
        contexualized_chunks.append(contextualized_chunk)
        
        print(f"--> {i}: original_chunk: {doc.page_content}")
        print(f"--> {i}: contexualized_chunk: {contextualized_chunk}")
        
        contexualized_docs.append(
            Document(
                page_content="\n"+contextualized_chunk+"\n\n"+doc.page_content,
                metadata=doc.metadata
            )
        )
    return contexualized_docs, contexualized_chunks

def add_to_opensearch(docs):    
    if len(docs) == 0:
        return []    
    #print('docs[0]: ', docs[0])       
            
    ids = []
    if enableParentDocumentRetrival == 'true':
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function = len,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            # separators=["\n\n", "\n", ".", " ", ""],
            length_function = len,
        )

        splitted_docs = parent_splitter.split_documents(docs)
        print('len(splitted_docs): ', len(splitted_docs))

        print('splitted_docs[0]: ', splitted_docs[0].page_content)

        parent_docs = []
        if contextual_embedding == 'Enable':
            parent_docs, contexualized_chunks = get_contextual_docs_from_chunks(docs[-1], splitted_docs)

            print('parent contextual chunk[0]: ', parent_docs[0].page_content)    
        else:
            parent_docs = splitted_docs  

        if len(parent_docs):
            for i, doc in enumerate(parent_docs):
                doc.metadata["doc_level"] = "parent"
                # print(f"parent_docs[{i}]: {doc}")
            print('parent_docs[0]: ', parent_docs[0].page_content)
                    
            try:
                parent_doc_ids = vectorstore.add_documents(parent_docs, bulk_size = 10000)
                print('parent_doc_ids: ', parent_doc_ids)
                ids = parent_doc_ids

                for i, doc in enumerate(splitted_docs):
                    _id = parent_doc_ids[i]
                    child_docs = child_splitter.split_documents([doc])
                    for _doc in child_docs:
                        _doc.metadata["parent_doc_id"] = _id
                        _doc.metadata["doc_level"] = "child"

                    if contextual_embedding == 'Enable':
                        contexualized_child_docs = [] # contexualized child doc
                        for _doc in child_docs:
                            contexualized_child_docs.append(
                                Document(
                                    page_content=contexualized_chunks[i]+"\n\n"+_doc.page_content,
                                    metadata=_doc.metadata
                                )
                            )
                        child_docs = contexualized_child_docs

                    print('child_docs[0]: ', child_docs[0].page_content)
                
                    child_doc_ids = vectorstore.add_documents(child_docs, bulk_size = 10000)
                    print('child_doc_ids: ', child_doc_ids)
                    print('len(child_doc_ids): ', len(child_doc_ids))
                        
                    ids += child_doc_ids
            except Exception:
                err_msg = traceback.format_exc()
                print('error message: ', err_msg)                
                #raise Exception ("Not able to add docs in opensearch")                
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function = len,
        ) 
        
        splitted_docs = text_splitter.split_documents(docs)
        print('len(splitted_docs): ', len(splitted_docs))

        if len(splitted_docs):
            if contextual_embedding == 'Enable':
                documents, contexualized_chunks = get_contextual_docs_from_chunks(docs[-1], splitted_docs)

                print('contextual chunks[0]: ', contexualized_chunks[0])  
            else:
                print('documents[0]: ', documents[0])
            
        try:        
            ids = vectorstore.add_documents(documents, bulk_size = 10000)
            print('response of adding documents: ', ids)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)
            #raise Exception ("Not able to add docs in opensearch")    
    return ids

def get_contextual_text(whole_text, splitted_text, llm): # per page
    contextual_template = (
        "<document>"
        "{WHOLE_DOCUMENT}"
        "</document>"
        "Here is the chunk we want to situate within the whole document."
        "<chunk>"
        "{CHUNK_CONTENT}"
        "</chunk>"
        "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk."
        "Answer only with the succinct context and nothing else in English."
        "Put it in <result> tags."
    )          
    
    contextual_prompt = ChatPromptTemplate([
        ('human', contextual_template)
    ])

    contextual_text = ""    
    
    contexual_chain = contextual_prompt | llm            
    response = contexual_chain.invoke(
        {
            "WHOLE_DOCUMENT": whole_text,
            "CHUNK_CONTENT": splitted_text
        }
    )    
    # print('--> contexual rext: ', response)
    output = response.content
    contextual_text = output[output.find('<result>')+8:output.find('</result>')]
    
    # print(f"--> whole_text: {whole_text}")
    print(f"--> original_chunk: {splitted_text}")
    print(f"--> contextual_text: {contextual_text}")

    return contextual_text

def delete_document_if_exist(metadata_key):
    try: 
        s3r = boto3.resource("s3")
        bucket = s3r.Bucket(s3_bucket)
        objs = list(bucket.objects.filter(Prefix=metadata_key))
        print('objs: ', objs)
        
        if(len(objs)>0):
            doc = s3r.Object(s3_bucket, metadata_key)
            meta = doc.get()['Body'].read().decode('utf-8')
            print('meta: ', meta)
            
            ids = json.loads(meta)['ids']
            print('ids: ', ids)
            
            # delete ids
            result = vectorstore.delete(ids)
            print('delete ids in vectorstore: ', result)   
            
            # delete files 
            files = json.loads(meta)['files']
            print('files: ', files)
            
            for file in files:
                s3r.Object(s3_bucket, file).delete()
                print('delete file: ', file)
            
        else:
            print('no meta file: ', metadata_key)
            
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to create meta file")

s3r = boto3.resource("s3")
def delete_if_exist(bucket, key):
    try: 
        s3r = boto3.resource("s3")
        bucket = s3r.Bucket(bucket)
        objs = list(bucket.objects.filter(Prefix=key))
        print('objs: ', objs)
        
        # if(len(objs)>0):
        if(len(objs)>0):
            # delete the object            
            print(f"delete -> bucket: {bucket}, key: {key}")
            for object in bucket.objects.filter(Prefix=key):
                print('object: ', object)
                object.delete()
            
            # delete metadata of the object
            if key.rfind('/'):
                objectName = key[key.rfind(doc_prefix)+len(doc_prefix):]
            else:
                objectName = key
            print('objectName: ', objectName)
            metadata_key = meta_prefix+objectName+'.metadata.json'
            print('meta file name: ', metadata_key)    
            delete_document_if_exist(metadata_key)
            time.sleep(2)
            
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to create meta file")

def extract_page_images_from_pdf(key, pages, nImages, contents, texts):
    files = []
    for i, page in enumerate(pages):
        print('page: ', page)
        
        imgInfo = page.get_image_info()
        print(f"imgInfo[{i}]: {imgInfo}")         
        
        if ocr=="Enable":
            contexual_text = ""
            
            print('start contextual embedding for image.')
            llm = get_model()
            contexual_text = get_contextual_text(contents, texts[i], llm)
            contexual_text.replace('\n','')

            # save current pdf page to image 
            pixmap = page.get_pixmap(dpi=200)  # dpi=300
            #pixels = pixmap.tobytes() # output: jpg
            
            # convert to png
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            pixels = BytesIO()
            img.save(pixels, format='PNG')
            pixels.seek(0, 0)
                            
            # get path from key
            objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)])
            folder = 'captures/'+objectName+'/'
            print('folder: ', folder)
                    
            fname = 'img_'+key.split('/')[-1].split('.')[0]+f"_{i}"
            print('fname: ', fname)          

            encoded_contexual_text = ""  # s3 meta only allows ASCII format
            encoded_contexual_text = contexual_text.encode('ascii', 'ignore').decode('ascii')
            encoded_contexual_text = re.sub('[^A-Za-z]', ' ', encoded_contexual_text)
            print('encoded_contexual_text: ', encoded_contexual_text)

            image_key = folder+fname+'.png'

            delete_if_exist(s3_bucket, image_key)
                
            print('create an table: ', image_key)
            response = s3_client.put_object(
                Bucket=s3_bucket,
                Key=image_key,
                ContentType='image/png',
                Metadata = {     
                    "type": 'image',                           
                    "ext": 'png',
                    "page": str(i),
                    "contextual_embedding": contextual_embedding,
                    "multi_region": multi_region,
                    "model_name": model_name,
                    "contextual_text": encoded_contexual_text,
                    "ocr": ocr
                },
                Body=pixels
            )
            print('response: ', response)
                                            
            files.append(image_key)

    return files

# load documents from s3 for pdf and txt
def load_document(file_type, key):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, key)
    
    files = []
    contents = ""
    if file_type == 'pdf':
        Byte_contents = doc.get()['Body'].read()

        texts = []
        nImages = []
        try: 
            # pdf reader            
            reader = PdfReader(BytesIO(Byte_contents))
            print('pages: ', len(reader.pages))
            
            # extract text
            imgList = []
            for i, page in enumerate(reader.pages):
                print(f"page[{i}]: {page}")
                texts.append(page.extract_text())
                
            contents = '\n'.join(texts)
            
            pages = fitz.open(stream=Byte_contents, filetype='pdf')     
            
            image_files = extract_page_images_from_pdf(key, pages, nImages, contents, texts)

            for img in image_files:
                files.append(img)
                print(f"image file: {img}")
                                                        
        except Exception:
                err_msg = traceback.format_exc()
                print('err_msg: ', err_msg)
                # raise Exception ("Not able to load the pdf file")
                     
    elif file_type == 'txt' or file_type == 'md':       
        try:  
            contents = doc.get()['Body'].read().decode('utf-8')
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)        
            # raise Exception ("Not able to load the file")
    
    return contents, files

def store_document_for_opensearch(file_type, key):
    print('upload to opensearch: ', key) 
    contents, files = load_document(file_type, key)
    
    if len(contents) == 0:
        print('no contents: ', key)
        return [], files
    
    # contents = str(contents).replace("\n"," ") 
    print('length: ', len(contents))
    
    # text
    docs = []
    docs.append(Document(
        page_content=contents,
        metadata={
            'name': key,
            'url': path+parse.quote(key)
        }
    ))    
    print('docs: ', docs)

    ids = add_to_opensearch(docs)
    
    return ids, files

if __name__ =="__main__":
    print(f"###### main ######")

    key = doc_prefix+"error_code.pdf"
    file_type = "pdf"
    ids, files = store_document_for_opensearch(file_type, key)   
    
    print('response of adding documents: ', ids)
    print('files: ', files)