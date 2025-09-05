import logging
import sys
import utils
import os
import boto3
import json
import requests

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-config")

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")

config = utils.load_config()
logger.info(f"config: {config}")

projectName = config["projectName"] if "projectName" in config else "mcp-rag"
region = config["region"] if "region" in config else "us-west-2"
managed_opensearch_url = config["managed_opensearch_url"] if "managed_opensearch_url" in config else None
opensearch_username = config["opensearch_username"] if "opensearch_username" in config else None
opensearch_password = config["opensearch_password"] if "opensearch_password" in config else None

aws_region = config["region"] if "region" in config else "us-west-2"
workingDir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"workingDir: {workingDir}")

session = boto3.Session()
credentials = session.get_credentials()

mcp_user_config = {}        
gateway_url = ""
bearer_token = ""

def get_bearer_token_from_secret_manager(secret_name):
    try:
        session = boto3.Session()
        client = session.client('secretsmanager', region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        bearer_token_raw = response['SecretString']
        
        token_data = json.loads(bearer_token_raw)        
        if 'bearer_token' in token_data:
            bearer_token = token_data['bearer_token']
            return bearer_token
        else:
            logger.info("No bearer token found in secret manager")
            return None
    
    except Exception as e:
        logger.info(f"Error getting stored token: {e}")
        return None

def get_cognito_config(cognito_config):    
    user_pool_name = cognito_config.get('user_pool_name')
    user_pool_id = cognito_config.get('user_pool_id')
    if not user_pool_name:        
        user_pool_name = projectName + '-agentcore-user-pool'
        print(f"No user pool name found in config, using default user pool name: {user_pool_name}")
        cognito_config.setdefault('user_pool_name', user_pool_name)

        cognito_client = boto3.client('cognito-idp', region_name=region)
        response = cognito_client.list_user_pools(MaxResults=60)
        for pool in response['UserPools']:
            if pool['Name'] == user_pool_name:
                user_pool_id = pool['Id']
                print(f"Found cognito user pool: {user_pool_id}")
                cognito_config['user_pool_id'] = user_pool_id
                break

    client_name = cognito_config.get('client_name')
    if not client_name:        
        client_name = f"{projectName}-agentcore-client"
        print(f"No client name found in config, using default client name: {client_name}")
        cognito_config['client_name'] = client_name

    client_id = cognito_config.get('client_id')
    if not client_id:
        response = cognito_client.list_user_pool_clients(UserPoolId=user_pool_id)
        for client in response['UserPoolClients']:
            if client['ClientName'] == client_name:
                client_id = client['ClientId']
                print(f"Found cognito client: {client_id}")
                cognito_config['client_id'] = client_id     
                break

    username = cognito_config.get('test_username')
    password = cognito_config.get('test_password')
    if not username or not password:
        print("No test username found in config, using default username and password. Please check config.json and update the test username and password.")
        username = f"{projectName}-test-user@example.com"
        password = "TestPassword123!"        
        cognito_config['test_username'] = username
        cognito_config['test_password'] = password
    
    return cognito_config

# get config of cognito
cognito_config = config.get('cognito', {})
if not cognito_config:
    cognito_config = get_cognito_config(cognito_config)
    if 'cognito' not in config:
        config['cognito'] = {}
    config['cognito'].update(cognito_config)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

def retrieve_bearer_token(secret_name):
    secret_name = config['secret_name']
    bearer_token = get_bearer_token_from_secret_manager(secret_name)
    logger.info(f"Bearer token from secret manager: {bearer_token[:100] if bearer_token else 'None'}...")

    # verify bearer token
    try:
        client = boto3.client('cognito-idp', region_name=region)
        response = client.get_user(
            AccessToken=bearer_token
        )
        logger.info(f"response: {response}")

        username = response['Username']
        logger.info(f"Username: {username}")

    except Exception as e:
        logger.info(f"Error verifying bearer token: {e}")

        # Try to get fresh bearer token from Cognito
        logger.info("Error verifying bearer token, getting fresh bearer token from Cognito...")
        bearer_token = create_cognito_bearer_token(config)
        logger.info(f"Bearer token from cognito: {bearer_token[:100] if bearer_token else 'None'}...")
        
        if bearer_token:
            secret_name = config['secret_name']
            save_bearer_token(secret_name, bearer_token)
        else:
            logger.info("Failed to get bearer token from Cognito. Exiting.")
            return {}
        
    return bearer_token

def save_bearer_token(secret_name, bearer_token):
    try:        
        session = boto3.Session()
        client = session.client('secretsmanager', region_name=region)
        
        # Create secret value with bearer_key 
        secret_value = {
            "bearer_key": "mcp_server_bearer_token",
            "bearer_token": bearer_token
        }
        
        # Convert to JSON string
        secret_string = json.dumps(secret_value)
        
        # Check if secret already exists
        try:
            client.describe_secret(SecretId=secret_name)
            # Secret exists, update it
            client.put_secret_value(
                SecretId=secret_name,
                SecretString=secret_string
            )
            logger.info(f"Bearer token updated in secret manager with key: {secret_value['bearer_key']}")
        except client.exceptions.ResourceNotFoundException:
            # Secret doesn't exist, create it
            client.create_secret(
                Name=secret_name,
                SecretString=secret_string,
                Description="MCP Server Cognito credentials with bearer key and token"
            )
            logger.info(f"Bearer token created in secret manager with key: {secret_value['bearer_key']}")
            
    except Exception as e:
        logger.info(f"Error saving bearer token: {e}")
        # Continue execution even if saving fails

def create_cognito_bearer_token(config):
    """Get a fresh bearer token from Cognito"""
    try:
        cognito_config = config['cognito']
        region = cognito_config['region']
        username = cognito_config['test_username']
        password = cognito_config['test_password']

        client_name = cognito_config['client_name']
        cognito_client = boto3.client('cognito-idp', region_name=region)
        try:
            response = cognito_client.list_user_pools(MaxResults=10)
            for pool in response['UserPools']:
                logger.info(f"Existing User Pool found: {pool['Id']}")
                user_pool_id = pool['Id']

                client_response = cognito_client.list_user_pool_clients(UserPoolId=user_pool_id)
                for client in client_response['UserPoolClients']:
                    if client['ClientName'] == client_name:
                        client_id = client['ClientId']
                        logger.info(f"Existing App client found: {client_id}")

                        # Update config.json with client_id
                        try:
                            config['cognito']['client_id'] = client_id
                            config_file = "config.json"
                            with open(config_file, "w") as f:
                                json.dump(config, f, indent=2)
                            logger.info(f"Client ID updated in config.json: {client_id}")
                        except Exception as e:
                            logger.info(f"Warning: Failed to update config.json with client_id: {e}")
        except Exception as e:
            logger.error(f"Failed to check User Pool list: {e}")
    
        # Create Cognito client
        client = boto3.client('cognito-idp', region_name=region)
        
        # Authenticate and get tokens
        response = client.initiate_auth(
            ClientId=client_id,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': username,
                'PASSWORD': password
            }
        )
        
        auth_result = response['AuthenticationResult']
        access_token = auth_result['AccessToken']
        # id_token = auth_result['IdToken']
        
        logger.info("Successfully obtained fresh Cognito tokens")
        return access_token
        
    except Exception as e:
        logger.info(f"Error getting Cognito token: {e}")
        return None

mcp_user_config = {}    

def get_agent_runtime_arn(mcp_type: str):
    #logger.info(f"mcp_type: {mcp_type}")
    agent_runtime_name = f"{projectName.lower()}_{mcp_type.replace('-', '_')}"
    logger.info(f"agent_runtime_name: {agent_runtime_name}")
    client = boto3.client('bedrock-agentcore-control', region_name=region)
    response = client.list_agent_runtimes(
        maxResults=100
    )
    logger.info(f"response: {response}")
    
    agentRuntimes = response['agentRuntimes']
    for agentRuntime in agentRuntimes:
        if agentRuntime["agentRuntimeName"] == agent_runtime_name:
            logger.info(f"agent_runtime_name: {agent_runtime_name}, agentRuntimeArn: {agentRuntime["agentRuntimeArn"]}")
            return agentRuntime["agentRuntimeArn"]
    return None

def get_gateway_url():
    gateway_client = boto3.client('bedrock-agentcore-control', region_name=region)
    response = gateway_client.list_gateways(maxResults=60)
    gateway_name = config['projectName']
    for gateway in response['items']:
        if gateway['name'] == gateway_name:
            print(f"gateway: {gateway}")
            gateway_id = gateway.get('gatewayId')
            config['gateway_id'] = gateway_id
            break
    gateway_url = f'https://{gateway_id}.gateway.bedrock-agentcore.{region}.amazonaws.com/mcp'
    logger.info(f"gateway_url: {gateway_url}")

    return gateway_url

def load_config(mcp_type):
    global bearer_token, gateway_url

    if mcp_type == "MCP Lambda (Knowledge Base)":
        mcp_type = 'knowledge_base_lambda'
    elif mcp_type == "AWS MCP (Knowledge Base)":
        mcp_type = 'knowledge_base_custom'
    elif mcp_type == "MCP Lambda (OpenSearch)":
        mcp_type = 'openSearch_lambda'
    elif mcp_type == "OpenSearch MCP":
        mcp_type = 'OpenSearch'
    elif mcp_type == "MCP Knowledge Base Retriever":
        mcp_type = 'mcp_server_retrieve'

    logger.info(f"mcp_type: {mcp_type}")

    if mcp_type == "Basic":
        return {
            "mcpServers": {
                "search": {
                    "command": "python",
                    "args": [
                        f"{workingDir}/mcp_server_basic.py"
                    ]
                }
            }
        }
    
    elif mcp_type == "knowledge_base_lambda":
        return {
            "mcpServers": {
                "knowledge_base_lambda": {
                    "command": "python",
                    "args": [
                        f"{workingDir}/mcp_server_lambda_knowledge_base.py"
                    ]
                }
            }
        }    
    
    elif mcp_type == "knowledge_base_custom":
        return {
            "mcpServers": {
                "knowledge_base_custom": {
                    "command": "python",
                    "args": [
                        f"{workingDir}/mcp_server_knowledge_base.py"
                    ],
                    "env": {
                        "KB_INCLUSION_TAG_KEY": "mcp-rag"
                    }
                }
            }
        }
    
    elif mcp_type == "openSearch_lambda":
        return {
            "mcpServers": {
                "openSearch_lambda": {
                    "command": "python",
                    "args": [
                        f"{workingDir}/mcp_server_lambda_opensearch.py"
                    ]
                }
            }
        }

    elif mcp_type == "OpenSearch":
        return {
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
    
    elif mcp_type == "mcp_server_retrieve":
        return {
            "mcpServers": {
                "mcp_server_retrieve": {
                    "command": "python",
                    "args": [f"{workingDir}/mcp_server_retrieve.py"]
                }
            }
        }
    elif mcp_type == "agentcore gateway":                    
        bearer_token = retrieve_bearer_token(config['secret_name'])
        if not gateway_url:            
            gateway_url = get_gateway_url()

        return {
            "mcpServers": {
                "agentcore-gateway": {
                    "type": "streamable_http",
                    "url": gateway_url,
                    "headers": {
                        "Authorization": f"Bearer {bearer_token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream"
                    }
                }
            }
        }

    elif mcp_type == "사용자 설정":
        return mcp_user_config

def load_selected_config(mcp_servers: dict):
    logger.info(f"mcp_servers: {mcp_servers}")
    
    loaded_config = {}
    for server in mcp_servers:
        logger.info(f"server: {server}")

        config = load_config(server)
        # logger.info(f"config: {config}")
        
        if config:
            loaded_config.update(config["mcpServers"])
    # logger.info(f"loaded_config: {loaded_config}")        
    return {
        "mcpServers": loaded_config
    }
