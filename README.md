# Advanced RAG

여기서는 Advanced RAG를 이용해 RAG 및 Agentic RAG를 구성하는것을 설명합니다. 또한 RAG는 MCP 서버를 이용해 편리하게 이용할 수 있습니다.

## OCR

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

## Parent Child Chunking

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

## MCP로 RAG 활용하기

### Knowledge Base 활용

완전 관리하여 RAG 서비스인 knowledge base는 S3와 같은 storage에 대해서 sync 기능을 제공하므로써 편리하게 사용할 수 있습니다. 하지만 OCR이나 contextual embedding을 이용할 경우에는 custom으로 lambda등을 활용하여 직접 파싱후 넣어주여야 합니다. 

MCP로 knowledge base를 조회하기 위해서, lambda를 이용하여 MCP 서버를 정의하거나, awslabs에서 제공하는 MCP 서버를 활용할 수 있습니다. lambda를 이용해 knowledge base를 조회하는 것은 [lambda-knowledge-base](./lambda-knowledge-base/lambda_function.py)을 참조합니다. 이를 [mcp_server_lambda_knowledge_base.py](./application/mcp_server_lambda_knowledge_base.py)와 같이 custom MCP 서버를 정의해 사용할 수 있습니다.

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

여기서 [mcp_knowledge_base.py](./appliccation/mcp_knowledge_base.py)와 같이 lambda로 직접 request를 보내서 아래와 같이 knowledge base의 문서들을 조회할 수 있습니다.

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

[AWS의 knowledge base MCP](https://awslabs.github.io/mcp/servers/bedrock-kb-retrieval-mcp-server/)를 활용하면 쉽게 연결하여 사용할 수 있습니다. mcp-tag를 이용해 관련된 knowledge base를 조회한 후에 query를 수행합니다. 

AWS의 knowledge base를 이용하면 별도로 인프라 없어 쉽게 조회가 가능하여 편리합니다. 반면에 Lambda로 MCP 서버를 구현할 경우에는 추가적인 인프라가 필요하지만, grading을 통해 관련도가 낮은 문서를 제외하는 것과 같은 custom 작업을 수행할 수 있고, knowledge base를 조회하지 않고 바로 query를 하므로 더 빠른 응답을 얻을 수 있습니다.




