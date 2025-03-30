"""
Cluster health agent for IronBox.
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


class ClusterHealthAgent:
    """Agent for checking Kubernetes cluster health."""
    
    def __init__(self, llm=default_llm, db_session: Optional[AsyncSession] = None):
        """
        Initialize ClusterHealthAgent.
        
        Args:
            llm: LLM instance
            db_session: Database session
        """
        self.llm = llm
        self.db_session = db_session
        self.system_prompt = """
        You are a Kubernetes cluster health agent for the IronBox system. Your job is to check the health of Kubernetes clusters and report any issues.
        
        You can check:
        - Pod status
        - Resource usage
        - Storage status (PVC/Volume health)
        
        Analyze the health check results and provide a clear summary of any issues found.
        """
    
    async def __call__(self, state: AgentState) -> AgentState:
        """
        Process a cluster health check request.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            # Check if we have a database session
            if not self.db_session:
                state.error = "Database session not available"
                state.agent_outputs[AgentType.CLUSTER_HEALTH] = {
                    "response": "I'm unable to check cluster health at the moment due to a database connection issue.",
                    "next": None,
                    "error": state.error,
                }
                return state
            
            # Extract cluster name from user input
            cluster_name = await self._extract_cluster_name(state.input)
            
            if not cluster_name:
                # No cluster specified, list available clusters
                clusters = await ClusterOperations.get_clusters(self.db_session)
                
                if not clusters:
                    state.agent_outputs[AgentType.CLUSTER_HEALTH] = {
                        "response": "No clusters are registered. Please register a cluster first.",
                        "next": None,
                    }
                    return state
                
                response = "Which cluster would you like to check? Here are the registered clusters:\n\n"
                for cluster in clusters:
                    response += f"- {cluster.name} ({cluster.api_server})\n"
                
                state.agent_outputs[AgentType.CLUSTER_HEALTH] = {
                    "response": response,
                    "next": None,
                }
                return state
            
            # Get the cluster
            cluster = await ClusterOperations.get_cluster_by_name(self.db_session, cluster_name)
            
            if not cluster:
                state.agent_outputs[AgentType.CLUSTER_HEALTH] = {
                    "response": f"I couldn't find a cluster named '{cluster_name}'. Please check the name and try again.",
                    "next": None,
                }
                return state
            
            # Connect to the cluster
            client = KubernetesClient(
                api_server=cluster.api_server,
                token=cluster.token,
                certificate=cluster.certificate,
                config_file=cluster.config_file,
                config_context=cluster.config_context,
                insecure_skip_tls_verify=cluster.insecure_skip_tls_verify,
            )
            
            connection_successful = client.connect()
            
            if not connection_successful:
                state.agent_outputs[AgentType.CLUSTER_HEALTH] = {
                    "response": f"I couldn't connect to the cluster '{cluster.name}' at {cluster.api_server}. Please check the connection details and try again.",
                    "next": None,
                }
                return state
            
            # Get health check
            health_check = client.get_health_check()
            
            # Save health check to database
            await ClusterOperations.add_health_check(
                self.db_session,
                cluster.id,
                {
                    "status": health_check["status"],
                    "pod_status": health_check["pod_status"],
                    "resource_usage": health_check["resource_usage"],
                    "storage_status": health_check["storage_status"],
                    "issues": health_check["issues"],
                }
            )
            
            # Generate response
            response = await self._generate_health_report(cluster.name, health_check)
            
            state.agent_outputs[AgentType.CLUSTER_HEALTH] = {
                "response": response,
                "next": None,
            }
            
            return state
        except Exception as e:
            logger.error(f"Error in cluster health agent: {e}")
            state.error = f"Cluster health agent error: {str(e)}"
            state.agent_outputs[AgentType.CLUSTER_HEALTH] = {
                "response": "An error occurred while checking cluster health.",
                "next": None,
                "error": state.error,
            }
            return state
    
    async def _extract_cluster_name(self, user_input: str) -> Optional[str]:
        """
        Extract cluster name from user input.
        
        Args:
            user_input: User input text
            
        Returns:
            Cluster name or None if not found
        """
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": """
            Extract the Kubernetes cluster name from the user's message. The user is asking to check the health of a cluster.
            
            Return only the cluster name, or "None" if no specific cluster is mentioned.
            """},
            {"role": "user", "content": user_input},
        ]
        
        # Get response from LLM
        response = await self.llm.ainvoke(messages)
        
        # Extract response text
        if hasattr(response, 'generations'):
            # ChatResult object
            response_text = response.generations[0].message.content.strip()
        elif isinstance(response, AIMessage):
            # AIMessage object
            response_text = response.content.strip()
        else:
            # Fallback
            response_text = str(response).strip()
        
        if response_text.lower() == "none":
            return None
        
        return response_text
    
    async def _generate_health_report(self, cluster_name: str, health_check: Dict[str, Any]) -> str:
        """
        Generate a health report from health check results.
        
        Args:
            cluster_name: Cluster name
            health_check: Health check results
            
        Returns:
            Health report text
        """
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": """
            You are a Kubernetes cluster health analyst. Generate a clear, concise health report for a Kubernetes cluster based on the health check results.
            
            Focus on:
            1. Overall health status
            2. Any critical or warning issues
            3. Pod status
            4. Resource usage
            5. Storage health
            
            Use formatting to make the report easy to read. Use bullet points for lists of issues.
            """},
            {"role": "user", "content": f"Generate a health report for cluster '{cluster_name}' based on these health check results:\n\n{health_check}"},
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
