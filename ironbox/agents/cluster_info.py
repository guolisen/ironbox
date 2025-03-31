"""
Cluster info agent for IronBox.
"""
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union

from sqlalchemy.ext.asyncio import AsyncSession
from langchain.schema import AIMessage

from ironbox.core.graph import AgentState, AgentType
from ironbox.util.kubernetes_client import KubernetesClient
from ironbox.core.llm import default_llm
from ironbox.db.operations import ClusterOperations

# Configure logging
logger = logging.getLogger(__name__)


class ClusterInfoAgent:
    """Agent for retrieving Kubernetes cluster information."""
    
    def __init__(self, llm=default_llm, db_session: Optional[AsyncSession] = None):
        """
        Initialize ClusterInfoAgent.
        
        Args:
            llm: LLM instance
            db_session: Database session
        """
        self.llm = llm
        self.db_session = db_session
        self.system_prompt = """
        You are a Kubernetes cluster information agent for the IronBox system. Your job is to provide information about registered Kubernetes clusters.
        
        You can provide:
        - List of registered clusters
        - Detailed information about a specific cluster
        - Cluster health history
        
        Extract the user's request and provide the requested information.
        """
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        Process a cluster information request.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            # Check if we have a database session
            if not self.db_session:
                state.error = "Database session not available"
                state.agent_outputs[AgentType.CLUSTER_INFO] = {
                    "response": "I'm unable to retrieve cluster information at the moment due to a database connection issue.",
                    "next": None,
                    "error": state.error,
                }
                return state
            
            # Determine the type of request
            request_type, cluster_name = await self._analyze_request(state.input)
            
            if request_type == "list":
                # List all clusters
                response = await self._list_clusters()
            elif request_type == "detail" and cluster_name:
                # Get details for a specific cluster
                response = await self._get_cluster_details(cluster_name)
            elif request_type == "health" and cluster_name:
                # Get health history for a specific cluster
                response = await self._get_health_history(cluster_name)
            else:
                # Unknown request
                response = "I'm not sure what information you're looking for. You can ask for a list of clusters, details about a specific cluster, or health history for a cluster."
            
            state.agent_outputs[AgentType.CLUSTER_INFO] = {
                "response": response,
                "next": None,
            }
            
            return state
        except Exception as e:
            logger.error(f"Error in cluster info agent: {e}")
            state.error = f"Cluster info agent error: {str(e)}"
            state.agent_outputs[AgentType.CLUSTER_INFO] = {
                "response": "An error occurred while retrieving cluster information.",
                "next": None,
                "error": state.error,
            }
            return state
    
    async def _analyze_request(self, user_input: str) -> Tuple[str, Optional[str]]:
        """
        Analyze the user's request to determine the type and cluster name.
        
        Args:
            user_input: User input text
            
        Returns:
            Tuple of (request_type, cluster_name)
        """
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": """
            Analyze the user's request for Kubernetes cluster information. Determine:
            1. The type of request: "list" (list all clusters), "detail" (details about a specific cluster), or "health" (health history for a specific cluster)
            2. The cluster name (if applicable)
            
            Return a JSON object with the following fields:
            - request_type: "list", "detail", or "health"
            - cluster_name: The name of the cluster (or null if not applicable)
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
                data = json.loads(json_match.group(1))
                return data.get("request_type", "unknown"), data.get("cluster_name")
            
            # If no JSON block, try to parse the entire response
            import json
            data = json.loads(response_text)
            return data.get("request_type", "unknown"), data.get("cluster_name")
        except Exception as e:
            logger.error(f"Error analyzing request: {e}")
            
            # Fallback to simple text analysis
            if "list" in user_input.lower() or "all" in user_input.lower():
                return "list", None
            
            # Look for cluster name
            cluster_name_match = re.search(r'cluster[:\s]+(["\']?)([^"\']+)\1', user_input, re.IGNORECASE)
            if cluster_name_match:
                cluster_name = cluster_name_match.group(2)
                if "health" in user_input.lower():
                    return "health", cluster_name
                return "detail", cluster_name
            
            return "unknown", None
    
    async def _list_clusters(self) -> str:
        """
        List all registered clusters.
        
        Returns:
            Response text
        """
        clusters = await ClusterOperations.get_clusters(self.db_session)
        
        if not clusters:
            return "No clusters are registered. You can register a cluster using the cluster registration feature."
        
        response = "# Registered Clusters\n\n"
        for cluster in clusters:
            status_emoji = "🟢" if cluster.health_status == "healthy" else "🟠" if cluster.health_status == "warning" else "🔴" if cluster.health_status == "critical" else "⚪"
            response += f"{status_emoji} **{cluster.name}**\n"
            response += f"   - API Server: {cluster.api_server}\n"
            response += f"   - Status: {cluster.health_status or 'Unknown'}\n"
            response += f"   - Last Check: {cluster.last_health_check.isoformat() if cluster.last_health_check else 'Never'}\n\n"
        
        response += "You can get more details about a specific cluster by asking for it by name."
        
        return response
    
    async def _get_cluster_details(self, cluster_name: str) -> str:
        """
        Get details for a specific cluster.
        
        Args:
            cluster_name: Cluster name
            
        Returns:
            Response text
        """
        cluster = await ClusterOperations.get_cluster_by_name(self.db_session, cluster_name)
        
        if not cluster:
            return f"I couldn't find a cluster named '{cluster_name}'. Please check the name and try again."
        
        # Connect to the cluster to get live information
        client = KubernetesClient(
            api_server=cluster.api_server,
            token=cluster.token,
            certificate=cluster.certificate,
            config_file=cluster.config_file,
            config_context=cluster.config_context,
            insecure_skip_tls_verify=cluster.insecure_skip_tls_verify,
        )
        
        connection_successful = client.connect()
        
        response = f"# Cluster: {cluster.name}\n\n"
        response += f"**API Server:** {cluster.api_server}\n"
        response += f"**Description:** {cluster.description or 'No description'}\n"
        response += f"**Health Status:** {cluster.health_status or 'Unknown'}\n"
        response += f"**Last Health Check:** {cluster.last_health_check.isoformat() if cluster.last_health_check else 'Never'}\n"
        response += f"**Created:** {cluster.created_at.isoformat()}\n"
        response += f"**Updated:** {cluster.updated_at.isoformat()}\n\n"
        
        if connection_successful:
            # Get live information
            cluster_info = client.get_cluster_info()
            
            response += "## Live Information\n\n"
            
            if cluster_info.get("version"):
                version = cluster_info["version"]
                response += f"**Kubernetes Version:** {version.get('git_version', 'Unknown')}\n"
                response += f"**Platform:** {version.get('platform', 'Unknown')}\n"
            
            response += f"**Nodes:** {cluster_info.get('nodes', 'Unknown')}\n"
            response += f"**Namespaces:** {cluster_info.get('namespaces', 'Unknown')}\n\n"
            
            if cluster_info.get("node_info"):
                response += "### Nodes\n\n"
                for node in cluster_info["node_info"]:
                    response += f"- **{node['name']}**\n"
                    response += f"  - Status: {node['status']}\n"
                    response += f"  - Kubelet Version: {node['kubelet_version']}\n"
                    response += f"  - OS: {node['os_image']}\n"
                    response += f"  - Architecture: {node['architecture']}\n\n"
        else:
            response += "⚠️ **Warning:** I couldn't connect to the cluster to get live information.\n\n"
        
        response += "You can check the health of this cluster using the cluster health check feature."
        
        return response
    
    async def _get_health_history(self, cluster_name: str) -> str:
        """
        Get health history for a specific cluster.
        
        Args:
            cluster_name: Cluster name
            
        Returns:
            Response text
        """
        cluster = await ClusterOperations.get_cluster_by_name(self.db_session, cluster_name)
        
        if not cluster:
            return f"I couldn't find a cluster named '{cluster_name}'. Please check the name and try again."
        
        health_checks = await ClusterOperations.get_health_checks(self.db_session, cluster.id, limit=5)
        
        if not health_checks:
            return f"No health checks have been performed for cluster '{cluster_name}'. You can check the health using the cluster health check feature."
        
        response = f"# Health History for {cluster.name}\n\n"
        
        for i, check in enumerate(health_checks):
            status_emoji = "🟢" if check.status == "healthy" else "🟠" if check.status == "warning" else "🔴" if check.status == "critical" else "⚪"
            response += f"## {status_emoji} Check {i+1}: {check.timestamp.isoformat()}\n\n"
            response += f"**Status:** {check.status}\n"
            
            if check.issues:
                response += "\n**Issues:**\n\n"
                for issue in check.issues:
                    severity_emoji = "🔴" if issue.get("severity") == "critical" else "🟠" if issue.get("severity") == "warning" else "⚪"
                    response += f"- {severity_emoji} **{issue.get('component', 'Unknown')}:** {issue.get('message', 'No message')}\n"
            else:
                response += "\n**No issues detected.**\n"
            
            response += "\n---\n\n"
        
        response += "You can perform a new health check using the cluster health check feature."
        
        return response
