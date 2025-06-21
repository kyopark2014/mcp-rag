FROM amazon/aws-lambda-python:3.13

WORKDIR /var/task

# Copy requirements first for better caching
COPY requirements.txt /var/task/

# Install dependencies with specific versions and upgrade pip first
RUN /var/lang/bin/python3 -m pip install --upgrade pip
RUN /var/lang/bin/python3 -m pip install -r requirements.txt

# Install additional packages that might not be in requirements.txt
RUN /var/lang/bin/python3 -m pip install PyMuPDF
RUN /var/lang/bin/python3 -m pip install pycryptodome

# Copy function code
COPY lambda_function.py /var/task/

CMD ["lambda_function.lambda_handler"]