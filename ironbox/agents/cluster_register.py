"""
Cluster register agent for IronBox.
"""
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union

from sqlalchemy.ext.asyncio import AsyncSession
from langchain.schema import AIMessage

from ironbox.core.graph import AgentState, AgentType
from ironbox.core.kubernetes_client import KubernetesClient
from ironbox.core.llm import default_llm
from ironbox.db.operations import ClusterOperations

# Configure logging
logger = logging.getLogger(__name__)


class ClusterRegisterAgent:
    """Agent for registering Kubernetes clusters."""
    
    def __init__(self, llm=default_llm, db_session: Optional[AsyncSession] = None):
        """
        Initialize ClusterRegisterAgent.
        
        Args:
            llm: LLM instance
            db_session: Database session
        """
        self.llm = llm
        self.db_session = db_session
        self.system_prompt = """
        You are a Kubernetes cluster registration agent for the IronBox system. Your job is to help users register their Kubernetes clusters.
        
        To register a cluster, you need the following information:
        - Cluster name
        - API server URL
        - Authentication method (token, certificate, or config file)
        
        Extract this information from the user's request and register the cluster.
        """
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        Process a cluster registration request.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            # Check if we have a database session
            if not self.db_session:
                state.error = "Database session not available"
                state.agent_outputs[AgentType.CLUSTER_REGISTER] = {
                    "response": "I'm unable to register clusters at the moment due to a database connection issue.",
                    "next": None,
                    "error": state.error,
                }
                return state
            
            # Extract cluster information from user input
            cluster_info = await self._extract_cluster_info(state.input)
            
            # Validate cluster information
            if not cluster_info.get("name") or not cluster_info.get("api_server"):
                # Not enough information, ask for more
                response = await self._generate_info_request(state.input, cluster_info)
                state.agent_outputs[AgentType.CLUSTER_REGISTER] = {
                    "response": response,
                    "next": None,
                }
                return state
            
            # Check if cluster already exists
            existing_cluster = await ClusterOperations.get_cluster_by_name(
                self.db_session, cluster_info["name"]
            )
            
            if existing_cluster:
                state.agent_outputs[AgentType.CLUSTER_REGISTER] = {
                    "response": f"A cluster named '{cluster_info['name']}' is already registered. Please use a different name or update the existing cluster.",
                    "next": None,
                }
                return state
            
            # Test connection to cluster
            client = KubernetesClient(
                api_server=cluster_info["api_server"],
                token=cluster_info.get("token"),
                certificate=cluster_info.get("certificate"),
                config_file=cluster_info.get("config_file"),
                config_context=cluster_info.get("config_context"),
                insecure_skip_tls_verify=cluster_info.get("insecure_skip_tls_verify", False),
            )
            
            connection_successful = client.connect()
            
            if not connection_successful:
                state.agent_outputs[AgentType.CLUSTER_REGISTER] = {
                    "response": f"I couldn't connect to the cluster at {cluster_info['api_server']}. Please check the connection details and try again.",
                    "next": None,
                }
                return state
            
            # Register the cluster
            cluster = await ClusterOperations.create_cluster(self.db_session, cluster_info)
            
            # Get cluster info for response
            cluster_info_response = client.get_cluster_info()
            
            # Create response
            response = f"Successfully registered cluster '{cluster.name}'.\n\n"
            response += f"Cluster information:\n"
            response += f"- API Server: {cluster.api_server}\n"
            
            if cluster_info_response.get("version"):
                version = cluster_info_response["version"]
                response += f"- Kubernetes Version: {version.get('git_version', 'Unknown')}\n"
            
            if cluster_info_response.get("nodes"):
                response += f"- Nodes: {cluster_info_response['nodes']}\n"
            
            if cluster_info_response.get("namespaces"):
                response += f"- Namespaces: {cluster_info_response['namespaces']}\n"
            
            state.agent_outputs[AgentType.CLUSTER_REGISTER] = {
                "response": response,
                "next": None,
            }
            
            return state
        except Exception as e:
            logger.error(f"Error in cluster register agent: {e}")
            state.error = f"Cluster register agent error: {str(e)}"
            state.agent_outputs[AgentType.CLUSTER_REGISTER] = {
                "response": "An error occurred while registering the cluster.",
                "next": None,
                "error": state.error,
            }
            return state
    
    async def _extract_cluster_info(self, user_input: str) -> Dict[str, Any]:
        """
        Extract cluster information from user input.
        
        Args:
            user_input: User input text
            
        Returns:
            Dictionary of cluster information
        """
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": """
            Extract Kubernetes cluster information from the user's message. Return a JSON object with the following fields:
            - name: The name of the cluster
            - api_server: The API server URL
            - token: Authentication token (if provided)
            - certificate: CA certificate (if provided)
            - config_file: Path to kubeconfig file (if provided)
            - config_context: Kubeconfig context (if provided)
            - insecure_skip_tls_verify: Whether to skip TLS verification (default: false)
            - description: Optional description of the cluster
            
            Only include fields that are explicitly mentioned in the user's message.
            """},
            {"role": "user", "content": user_input},
        ]
        
        # Get response from LLM
        response = await self.llm.ainvoke(messages)
        
        # Extract response text
        if hasattr(response, 'generations'):
            # ChatResult object
            response_text = response.generations[0].message.content
        elif isinstance(response, AIMessage):
            # AIMessage object
            response_text = response.content
        else:
            # Fallback
            response_text = str(response)
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                import json
                return json.loads(json_match.group(1))
            
            # If no JSON block, try to parse the entire response
            import json
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Error extracting cluster info: {e}")
            
            # Fallback to regex extraction
            cluster_info = {}
            
            # Extract name
            name_match = re.search(r'name[:\s]+(["\']?)([^"\']+)\1', response_text)
            if name_match:
                cluster_info["name"] = name_match.group(2)
            
            # Extract API server
            api_server_match = re.search(r'api[_\s]*server[:\s]+(["\']?)([^"\']+)\1', response_text, re.IGNORECASE)
            if api_server_match:
                cluster_info["api_server"] = api_server_match.group(2)
            
            # Extract token
            token_match = re.search(r'token[:\s]+(["\']?)([^"\']+)\1', response_text)
            if token_match:
                cluster_info["token"] = token_match.group(2)
            
            return cluster_info
    
    async def _generate_info_request(self, user_input: str, partial_info: Dict[str, Any]) -> str:
        """
        Generate a request for more information.
        
        Args:
            user_input: User input text
            partial_info: Partial cluster information
            
        Returns:
            Response text
        """
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": """
            You are helping a user register a Kubernetes cluster. You have partial information and need to ask for the missing details.
            
            Required information:
            - name: The name of the cluster
            - api_server: The API server URL
            - Authentication method (one of):
              - token: Authentication token
              - certificate: CA certificate
              - config_file: Path to kubeconfig file
            
            Generate a response asking for the missing information. Be polite and helpful.
            """},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": f"I've extracted the following information:\n{partial_info}"},
        ]
        
        # Get response from LLM
        response = await self.llm.ainvoke(messages)
        
        # Extract response text
        if hasattr(response, 'generations'):
            # ChatResult object
            return response.generations[0].message.content
        elif isinstance(response, AIMessage):
            # AIMessage object
            return response.content
        else:
            # Fallback
            return str(response)
