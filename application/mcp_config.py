import chat
import logging
import sys
import utils
import boto3

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

session = boto3.Session()
credentials = session.get_credentials()

mcp_user_config = {}        
def load_config(mcp_type):
    if mcp_type == "basic":
        return {
            "mcpServers": {
                "search": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_basic.py"
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
                        "application/mcp_server_lambda_knowledge_base.py"
                    ]
                }
            }
        }    
    
    elif mcp_type == "aws_knowledge_base":  
        return {
            "mcpServers": {
                "aws_knowledge_base": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_kb.py"
                    ],
                    "env": {
                        "KB_INCLUSION_TAG_KEY": "mcp-rag"
                    }
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
            
    elif mcp_type == "사용자 설정":
        return mcp_user_config

def load_selected_config(mcp_selections: dict[str, bool]):
    #logger.info(f"mcp_selections: {mcp_selections}")
    loaded_config = {}

    selected_servers = [server for server, is_selected in mcp_selections.items() if is_selected]
    # logger.info(f"selected_servers: {selected_servers}")

    for server in selected_servers:
        # logger.info(f"server: {server}")

        if server == "image generation":
            config = load_config('image_generation')
        elif server == "aws diagram":
            config = load_config('aws_diagram')
        elif server == "aws document":
            config = load_config('aws_documentation')
        elif server == "aws cost":
            config = load_config('aws_cost')
        elif server == "ArXiv":
            config = load_config('arxiv')
        elif server == "aws cloudwatch":
            config = load_config('aws_cloudwatch')
        elif server == "aws storage":
            config = load_config('aws_storage')
        elif server == "Knowledge Base (Lambda)":
            config = load_config('knowledge_base_lambda')
        elif server == "code interpreter":
            config = load_config('code_interpreter')
        elif server == "aws cli":
            config = load_config('aws_cli')
        elif server == "text editor":
            config = load_config('text_editor')
        else:
            config = load_config(server)
        logger.info(f"config: {config}")
        
        if config:
            loaded_config.update(config["mcpServers"])

    # logger.info(f"loaded_config: {loaded_config}")
        
    return {
        "mcpServers": loaded_config
    }
