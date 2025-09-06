FROM amazon/aws-lambda-python:3.13

WORKDIR /var/task

COPY lambda_function.py /var/task/
COPY info.py /var/task/

RUN /var/lang/bin/python3 -m pip install langchain
RUN /var/lang/bin/python3 -m pip install langchain-community
RUN /var/lang/bin/python3 -m pip install langchain-aws
RUN /var/lang/bin/python3 -m pip install requests-aws4auth
RUN /var/lang/bin/python3 -m pip install opensearch-py

CMD ["lambda_function.lambda_handler"]