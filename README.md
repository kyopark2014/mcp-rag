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




