# MCP RAG

MCP를 이용하여 RAG를 편리하에 이용할 수 있습니다. 여기에서는 RAG의 성능향상 기법인 advanced RAG를 도입하고, 완전 관리형 RAG 서버스인 knowledge base와 관리형 RAG인 OpenSearch에서 MCP를 활용하는 방법을 설명합니다. 전체적인 architecture는 아래와 같습니다. 여기서는 MCP로 활용할 수 있는 4가지 형태의 RAG를 설명하고 있습니다. 사용자가 문서를 Amazon S3로 업로드하면 Knowledge Base에서는 sync를 통해 문서를 가져와서 Amazon Opensearch Serverless로 문서를 적재합니다. 이때 미리 지정한 embedding model을 이용하고 multi modal을 통해 분석된 정보를 활용할 수 있습니다. 또한 Amazon S3에 문서가 업로드 될때 발생하는 event를 AWS Lambda (s3-event-manager)가 받아서 SQS에 전달한 다음에 순차적으로 AWS Lambda (document-manager)가 embedding 및 multi modal 분석을 통해 얻어진 context를 Managed OpenSearch에 적재할 수 있습니다. Amazon EC2에 있는 AI application은 MCP client / server 구조를 이용해 MCP 서버의 tool들을 활용할 수 있습니다. 이때 RAG를 활용할 때에 아래 그림과 같이 (1) AWS MCP (Knowledge Base) (2) MCP Lambda (Knowledge Base) (3) OpenSearch MCP (4) MCP Lambda (OpenSearch)의 4가지 방법중에 한가지를 선택하여 활용할 수 있습니다. RAG를 이용해 필요한 OpenSearch, Lambda등의 인프라는 AWS CDK를 이용하여 쉽게 배포할 수 있습니다.

![image](https://github.com/user-attachments/assets/d1296cf3-af42-49e5-8d8a-fa4a2ca9150e)


OCR은 문서의 페이지들을 캡춰하여 이미지를 생성한 후에 OpenSearch에서 검색하므로, [Agentic RAG 구현하기](https://github.com/kyopark2014/agentic-rag)와 같이 event 형태로 데이터 처리 파이프라인을 만들어서 활용하여야 합니다. 이벤트 처리는 [lambda-s3-event-manager](./lambda-s3-event-manager/lambda_function.py)로 수행하고, 문서의 처리는 [lambda-document-manager](./lambda-document-manager/lambda_function.py)로 구현합니다.

![image](https://github.com/user-attachments/assets/d99c8420-fa45-4ad6-8b48-ffc3df1caa30)

## Advanced RAG 기법

RAG의 성능을 향상시키기 위한 advanced RAG 기법에 대해 설명합니다. 

### OCR

문서의 각 페이지를 이미지로 변환한 후에 multimodal을 통해 분석합니다. 이때 맥락에 맞는 이미지 분석을 위해 contextual embedding을 활용합니다. 상세한 코드는 [lambda-document-manager](./lambda-document-manager/lambda-document-manager.py)을 참조합니다.

Contextual embedding을 위해 managed OpenSearch를 활용합니다. 여기서 os_client는 아래와 같이 정의하고 OpenSearch index를 생성할때 이용합니다.

```python
session = boto3.Session(region_name=region)
credentials = session.get_credentials()

awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    'es',  
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
```

이제 vectorstore 정의해서 문서를 추가하거나 삭제할때에 활용합니다.

```python
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
vectorstore = OpenSearchVectorSearch(
    index_name=index_name,  
    is_aoss = False,
    embedding_function=bedrock_embeddings,
    opensearch_url=opensearch_url,
    http_auth=awsauth,
    connection_class=RequestsHttpConnection
)
```

각 페이지가 전체 문서에서 어떤 의미를 가지는지 contextual_text를 추출하여 활용합니다.

```python
def get_contextual_text(whole_text, splitted_text, llm): 
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
    output = response.content
    return output[output.find('<result>')+8:output.find('</result>')]
```

페이지 이미지에서 텍스트를 추출하기 위해 이미지 사이즈를 조정한 후에 contextual text와 함께 OpenSearch에 등록하여 활용합니다.

```python
def store_image_for_opensearch(key):
    image_obj = s3_client.get_object(Bucket=s3_bucket, Key=key)
    image_content = image_obj['Body'].read()
    img = Image.open(BytesIO(image_content))
                        
    width, height = img.size 
    print(f"width: {width}, height: {height}, size: {width*height}")
            
    isResized = False
    while(width*height > 5242880):
        width = int(width/2)
        height = int(height/2)
        isResized = True
        print(f"width: {width}, height: {height}, size: {width*height}")
           
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                                                            
    llm = get_model()
    text = extract_text(llm, img_base64)
    extracted_text = text[text.find('<result>')+8:text.find('</result>')] 
    
    contextual_text = object_meta["contextual_text"]
    summary = summary_image(llm, img_base64, contextual_text)
    image_summary = summary[summary.find('<result>')+8:summary.find('</result>')]    
    contents = f"[이미지 요약]\n{image_summary}\n\n[추출된 텍스트]\n{extracted_text}"
    
    page = object_meta["page"]
    docs = []
    docs.append(
        Document(
            page_content=contents,
            metadata={
                'name': key,
                'page': page,
                'url': path+parse.quote(key)
            }
        )
    )
    return add_to_opensearch(docs)                                                                                                      
```

여기서 multimodal을 이용해 이미지에서 텍스트를 추출하는 함수는 아래와 같습니다.

```python
def summary_image(llm, img_base64, contextual_text):  
    query = "이미지가 의미하는 내용을 풀어서 자세히 알려주세요. markdown 포맷으로 답변을 작성합니다."
    if contextual_text:
        query += "\n아래 <reference>는 이미지와 관련된 내용입니다. 이미지 분석시 참고하세요. \n<reference>\n"+contextual_text+"\n</reference>"    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]    
    result = llm.invoke(messages)
    extracted_text = result.content        
    return extracted_text
```

### Parent Child Chunking

문서 검색의 정확도를 높이면서 Context를 충분히 사용하기 위해서는 Parent Child Chunking을 적용하여야 합니다. 이를 위해 아래와 같이 RecursiveCharacterTextSplitter로 parent와 child를 각각 나눕니다.

```python
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
    length_function = len,
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function = len,
)
```

먼저 parent chunk들을 OpenSearch에 넣고 id들을 확인합니다. child chunk의 meta에 parent chunk의 id를 추가해서 검색시 child chunk를 하고, 실제 context는 parent의 id로 조회한 parent의 text를 활용합니다. Contextual text는 실제 사용하게 될 parent chunk의 특징을 설명하여야 하므로, parent chunk로 얻은 contextual text를 child chunk에 추가하여 활용합니다. 이후 child chunk들도 OpenSearch에 등록하여 id들을 확인합니다. parent/child의 id들을 파일 meta에 저장하였다가 문서 업데이트/삭제시에 활용합니다. 

```python
splitted_docs = parent_splitter.split_documents(docs)
parent_docs, contexualized_chunks = get_contextual_docs_using_parallel_processing(docs[-1], splitted_docs)

for i, doc in enumerate(parent_docs):
    doc.metadata["doc_level"] = "parent"
        
parent_doc_ids = vectorstore.add_documents(parent_docs, bulk_size = 10000)
ids = parent_doc_ids

for i, doc in enumerate(splitted_docs):
    _id = parent_doc_ids[i]
    child_docs = child_splitter.split_documents([doc])
    for _doc in child_docs:
        _doc.metadata["parent_doc_id"] = _id
        _doc.metadata["doc_level"] = "child"

    contexualized_child_docs = []
    for _doc in child_docs:
        contexualized_child_docs.append(
            Document(
                page_content=contexualized_chunks[i]+"\n\n"+_doc.page_content,
                metadata=_doc.metadata
            )
        )
    child_docs = contexualized_child_docs

    child_doc_ids = vectorstore.add_documents(child_docs, bulk_size = 10000)        
    ids += child_doc_ids           
```

### Knowledge Base 활용

완전 관리형 RAG 서비스인 knowledge base는 S3와 같은 storage에 대해서 sync 기능을 제공하므로써 편리하게 사용할 수 있습니다. 하지만 OCR이나 contextual embedding을 이용할 경우에는 custom으로 lambda등을 활용하여 직접 파싱후 넣어주여야 합니다. 

## MCP로 RAG 활용하기

### AWS MCP (Knowledge Base)

[Amazon Bedrock Knowledge Base Retrieval MCP Server](https://awslabs.github.io/mcp/servers/bedrock-kb-retrieval-mcp-server/)와 같이 AWS에서 제공하는 MCP를 이용하여 Amazon Knowledge Base의 문서를 조회할 수 있습니다. 이때 별도로 조회하는 인프라가 필요하지 않으므로 쉽게 구현이 가능합니다. 이때의 MCP 설정은 아래와 같습니다.

```python
{
  "mcpServers": {
    "awslabs.bedrock-kb-retrieval-mcp-server": {
      "command": "uvx",
      "args": ["awslabs.bedrock-kb-retrieval-mcp-server@latest"],
      "env": {
        "AWS_PROFILE": "your-profile-name",
        "AWS_REGION": "us-east-1",
        "FASTMCP_LOG_LEVEL": "ERROR",
        "KB_INCLUSION_TAG_KEY": "optional-tag-key-to-filter-kbs",
        "BEDROCK_KB_RERANKING_ENABLED": "false"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

[AWS의 knowledge base MCP](https://awslabs.github.io/mcp/servers/bedrock-kb-retrieval-mcp-server/)는 knowledge base를 조회하여 특정 tag(기본은 mcp-tag)를 knowledge base를 찾은 다음에 query를 수행합니다. 따라서 knowledge base를 생성할 때에 아래와 같이 tag를 설정하여야 합니다.

![image](https://github.com/user-attachments/assets/0f6944e3-6d46-4dcb-b258-83d468377a92)

[Amazon Bedrock Knowledge Base Retrieval MCP Server](https://github.com/awslabs/mcp/tree/main/src/bedrock-kb-retrieval-mcp-server)에서는 GetKnowledgeBases를 resource로 호출하는데, LangGraph에서 참조가 안되는 이슈가 있습니다. 따라서, 여기에서는 소스를 복사하여 [mcp_server_knowledge_base.py](./application/mcp_server_knowledge_base.py)와 같이 tool로 수정하여서 활용하고 있습니다. 이때의 MCP 설정은 아래와 같습니다. 

```java
{
    "mcpServers": {
        "knowledge_base_custom": {
            "command": "python",
            "args": [
                "application/mcp_server_knowledge_base.py"
            ],
            "env": {
                "KB_INCLUSION_TAG_KEY": "mcp-rag"
            }
        }
    }
}
```

### MCP Lambda (Knowledge Base)

MCP로 knowledge base를 조회하기 위해서, lambda를 이용하면 사용자의 목적에 맞는 RAG 동작을 구현할 수 있스니다. 아래에서는 Lambda를 이용한 custom MCP 서버를 정의하는것을 설명합니다. Lambda를 이용해 knowledge base를 조회하는 것은 [lambda-knowledge-base](./lambda-knowledge-base/lambda_function.py)에 관련된 코드가 있습니다. 아래와 같이 knowledge_base_search를 tool로 정의합니다.

```python
@mcp.tool()
def knowledge_base_search(keyword: str) -> list:
    """
    Search the knowledge base with the given keyword.
    keyword: the keyword to search
    return: the result of search
    """

    return rag.retrieve_knowledge_base(keyword)
```

knowledge_base_search는 [mcp_knowledge_base.py](./appliccation/mcp_knowledge_base.py)에 정의된 retrieve_knowledge_base와 같이 lambda를 직접 호출하는 방식으로 knowledge base의 문서들을 조회합니다.

```python
def retrieve_knowledge_base(query):
    lambda_client = boto3.client(
        service_name='lambda',
        region_name=bedrock_region
    )
    functionName = f"knowledge-base-for-{projectName}"

    mcp_env = utils.load_mcp_env()
    grading_mode = mcp_env['grading_mode']
    multi_region = mcp_env['multi_region']

    payload = {
        'function': 'search_rag',
        'knowledge_base_name': knowledge_base_name,
        'keyword': query,
        'top_k': numberOfDocs,
        'grading': grading_mode,
        'model_name': model_name,
        'multi_region': multi_region
    }

    output = lambda_client.invoke(
        FunctionName=functionName,
        Payload=json.dumps(payload),
    )
    payload = json.load(output['Payload'])        
    return payload['response']
```

Lambda로 MCP 서버를 구현하면 추가적인 인프라가 필요하지만, grading을 통해 관련도가 낮은 문서를 제외하는 것과 같은 custom 작업을 수행할 수 있고, knowledge base를 조회하지 않고 바로 query를 하므로 더 빠른 응답을 얻을 수 있습니다.

### OpenSearch MCP

OpenSearch MCP를 이용하면 추가적인 리소스 없이 바로 OpenSearch를 조회할 수 있습니다. 하지만 2025년 6월 현재는 텍스트 검색만을 제공하여 성능상 제한이 있습니다. OpenSearch MCP를 이용할 때에는 아래 config를 활용합니다.

```python
{
    "mcpServers": {
        "opensearch-mcp-server": {
            "command": "uvx",
            "args": [
                "opensearch-mcp-server-py"
            ],
            "env": {
                "OPENSEARCH_URL": managed_opensearch_url,
                "AWS_REGION": aws_region,
                "AWS_ACCESS_KEY_ID": credentials.access_key,
                "AWS_SECRET_ACCESS_KEY": credentials.secret_key
            }
        }
    }
}
```

### MCP Lambda (OpenSearch)

OCR, contextual embedding과 같은 custom한 RAG를 구현할 때에는 직접 OpenSearch에 넣고 조회하여야 합니다. 이를 조회할 때에는 lambda로 custom MCP 서버를 정의하거나 [OpenSearch MCP](https://opensearch.org/blog/introducing-mcp-in-opensearch/)을 사용할 수 있습니다. Lambda로 OpenSearch를 조회하는 것은 [lambda-opensearch](./lambda-opensearch/lambda_function.py)와 같이 구현할 수 있습니다. 이를 custom MCP 서버로 구현할 때에는 [mcp_server_lambda_opensearch.py](./application/mcp_server_lambda_opensearch.py)와 같이 구현합니다.

```python
@mcp.tool()
def opensearch_search(keyword: str) -> list:
    """
    Search the knowledge base with the given keyword.
    keyword: the keyword to search
    return: the result of search
    """

    return rag.retrieve_opensearch(keyword)
```

여기서 [mcp_opensearch.py](./application/mcp_opensearch.py)와 같이 [lambda-opensearch](./lambda-opensearch/lambda_function.py)로 직접 요청하여 Lambda가 가져온 OpenSearch 문서를 얻을 수 있습니다. Lambda에서는 OpenSearch로 문서 조회뿐 아니라, 문서의 관련도를 기준으로 grading을 수행할 수 있습니다.

```python
def retrieve_opensearch(query):
    lambda_client = boto3.client(
        service_name='lambda',
        region_name=bedrock_region
    )
    functionName = f"opensearch-for-{projectName}"

    mcp_env = utils.load_mcp_env()
    grading_mode = mcp_env['grading_mode']
    multi_region = mcp_env['multi_region']

    payload = {
        'function': 'search_rag',
        'keyword': query,
        'top_k': numberOfDocs,
        'grading': grading_mode,
        'model_name': model_name,
        'multi_region': multi_region
    }
    output = lambda_client.invoke(
        FunctionName=functionName,
        Payload=json.dumps(payload),
    )
    payload = json.load(output['Payload'])        
    return payload['response']
```


## 설치하기

Repository를 clone 합니다.

```text
git clone https://github.com/kyopark2014/mcp-rag/
```

필요한 라이브러리를 설치합니다. 

```python
cd mcp-rag && pip install -r requirements.txt
```

CDK로 구동이 필요한 인프라인 CloudFront, S3, OpenSearch, Knowledge base, tavily, weather등의 secret을 설치합니다. 만약 cdk boootstraping이 안되어 있다면 설치후 수행합니다.

```text
cd cdk-mcp-rag/ && cdk deploy --all
```

설치가 완료되면, 아래와 같이 "CdkMcpRagStack.environmentformcprag"를 복사하여 application/config.json 파일을 생성합니다.

![image](https://github.com/user-attachments/assets/386edb27-ed29-49df-9df1-447b457e70ec)

config.json은 agent의 동작에 필요한 정보를 가지고 있고, [.gitignore](./application/.gitignore)에 의해 git으로 공유 되지 않습니다. 생성된 config.json의 셈플은 아래와 같습니다.

```json
{
    "projectName":"mcp-rag",
    "accountId":"862926741992",
    "region":"us-west-2",
    "roleKnowledgeBase":"arn:aws:iam::862926741992:role/role-knowledge-base-for-mcp-rag-us-west-2",
    "collectionArn":"arn:aws:aoss:us-west-2:862926741992:collection/8krsnuq4it9gpl70i3u6",
    "serverless_opensearch_url":"https://8krsnuq4it9gpl70i3u6.us-west-2.aoss.amazonaws.com",
    "managed_opensearch_url":"https://search-mcp-rag-mxtkul3z3qv5iiqprb7q3jx4wy.us-west-2.es.amazonaws.com",
    "knowledge_base_role":"arn:aws:iam::862926741992:role/role-knowledge-base-for-mcp-rag-us-west-2",
    "s3_bucket":"storage-for-mcp-rag-862926741992-us-west-2",
    "s3_arn":"arn:aws:s3:::storage-for-mcp-rag-862926741992-us-west-2",
    "sharing_url":"https://d3mo4kqj5cjiuy.cloudfront.net"
 }
```

이후 [Secret Manager](https://us-west-2.console.aws.amazon.com/secretsmanager/listsecrets?region=us-west-2)에 접속하여 아래와 같은 credential을 입력합니다.

![image](https://github.com/user-attachments/assets/b4dbd00b-02ed-4962-8e8b-7090859acc88)

만약 streamlit이 설치되어 있지 않다면 [streamlit](https://docs.streamlit.io/get-started/installation)을 참조하여 설치합니다. 이후 아래와 같이 실행합니다.

```text
streamlit run application/app.py
```

실행하면 아래와 같은 화면이 보여집니다. Agent를 선택하면 실행하고 동작을 확인할 수 있습니다. 


## 실행 결과

여기에서는 아래와 같은 MCP 서버를 제공합니다.

![image](https://github.com/user-attachments/assets/dbee256b-339c-4292-bd5e-7924176b9c2f)

"AWS MCP (Knowledge Base)"을 선택하면, 아래와 같이 GetKnowledgeBases으로 mcp-rag라는 tag를 가진 knowledge base를 검색한 후에 QueryKnowledgeBases로 검색을 수행합니다. 

<img src="https://github.com/user-attachments/assets/5f0bd428-3286-4c85-a8bd-8846c8f2d94e" width="600">


"MCP Lambda (Knowledge Base)"를 선택하고 검색하여 아래와 같이 knowledge_base_search를 이용해 문서 검색한 결과를 얻을 수 있습니다.

<img src="https://github.com/user-attachments/assets/8c5963da-0999-4c83-9827-85d47f17391f" width="600">


"OpenSearch MCP"로 검색하면 OpenSearch MCP를 이용해 조회합니다. Text 검색이므로 아래와 같이 결과가 안나올 수도 있습니다. 

<img src="https://github.com/user-attachments/assets/ec2d7ea1-826d-4471-a5f8-23c5c3872328" width="600">


"OpenSearch MCP"에서 "보일러 코드?"와 같이 검색하면 아래와 같이 검색에 성공합니다.

<img src="https://github.com/user-attachments/assets/d9eb4b58-0156-4d41-901f-fa155270a1e0" width="600">


"MCP Lambda (OpenSearch)"로 검색하면, 아래와 같이 opensearch_search를 이용하여 OpenSearch를 검색한 결과를 얻을 수 있습니다.

<img src="https://github.com/user-attachments/assets/137e029b-5f9c-410c-a05a-cdcf86d6d307" width="600">

