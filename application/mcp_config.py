import logging
import sys
import utils
import boto3
import os

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-cost")

config = utils.load_config()
print(f"config: {config}")

managed_opensearch_url = config["managed_opensearch_url"] if "managed_opensearch_url" in config else None
opensearch_username = config["opensearch_username"] if "opensearch_username" in config else None
opensearch_password = config["opensearch_password"] if "opensearch_password" in config else None

aws_region = config["region"] if "region" in config else "us-west-2"
workingDir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"workingDir: {workingDir}")

session = boto3.Session()
credentials = session.get_credentials()

mcp_user_config = {}        
def load_config(mcp_type):
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
