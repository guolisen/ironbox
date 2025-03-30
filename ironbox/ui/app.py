"""
Streamlit UI for IronBox.
"""
import logging
import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union

import streamlit as st
import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ironbox.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API URL
API_URL = f"http://{config['api']['host']}:{config['api']['port']}"


# Helper functions
async def api_request(method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Make an API request.
    
    Args:
        method: HTTP method
        endpoint: API endpoint
        data: Request data
        
    Returns:
        Response data
    """
    url = f"{API_URL}{endpoint}"
    
    # Increase timeout for chat endpoint which may take longer
    timeout = 60.0 if endpoint == "/chat" else 10.0
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        if method.lower() == "get":
            response = await client.get(url)
        elif method.lower() == "post":
            response = await client.post(url, json=data)
        elif method.lower() == "delete":
            response = await client.delete(url)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        if response.status_code >= 400:
            st.error(f"API Error: {response.text}")
            return {"error": response.text}
        
        return response.json()


def run_async(func):
    """
    Run an async function.
    
    Args:
        func: Async function
        
    Returns:
        Function result
    """
    return asyncio.run(func)


# Page functions
def chat_page():
    """Chat page."""
    st.title("IronBox Chat")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("How can I help you manage your Kubernetes clusters?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = run_async(
                    api_request(
                        "post",
                        "/chat",
                        {
                            "message": prompt,
                            "session_id": st.session_state.session_id,
                        },
                    )
                )
                
                if "error" in response:
                    st.error(response["error"])
                    return
                
                st.markdown(response["response"])
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response["response"]})


def clusters_page():
    """Clusters page."""
    st.title("Kubernetes Clusters")
    
    # Get clusters
    clusters = run_async(api_request("get", "/clusters"))
    
    if "error" in clusters:
        st.error(clusters["error"])
        return
    
    # Display clusters
    if not clusters:
        st.info("No clusters registered. Use the form below to register a new cluster.")
    else:
        # Create a DataFrame for display
        df = pd.DataFrame(clusters)
        df["health_status"] = df["health_status"].fillna("Unknown")
        
        # Add status emoji
        def status_emoji(status):
            if status == "healthy":
                return "🟢"
            elif status == "warning":
                return "🟠"
            elif status == "critical":
                return "🔴"
            else:
                return "⚪"
        
        df["status_emoji"] = df["health_status"].apply(status_emoji)
        
        # Display as a table
        st.subheader("Registered Clusters")
        
        # Custom display
        for _, row in df.iterrows():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.markdown(f"### {row['status_emoji']} {row['name']}")
            
            with col2:
                st.markdown(f"**API Server:** {row['api_server']}")
                st.markdown(f"**Status:** {row['health_status']}")
                if row["description"]:
                    st.markdown(f"**Description:** {row['description']}")
            
            with col3:
                if st.button("View Details", key=f"view_{row['id']}"):
                    st.session_state.selected_cluster = row["id"]
                    st.session_state.page = "cluster_details"
                    st.rerun()
                
                if st.button("Check Health", key=f"health_{row['id']}"):
                    st.session_state.selected_cluster = row["id"]
                    st.session_state.page = "cluster_health"
                    st.rerun()
                
                if st.button("Delete", key=f"delete_{row['id']}"):
                    if st.session_state.get("confirm_delete") == row["id"]:
                        # Confirmed delete
                        result = run_async(api_request("delete", f"/clusters/{row['id']}"))
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.success(result["message"])
                            st.session_state.pop("confirm_delete", None)
                            st.rerun()
                    else:
                        # Ask for confirmation
                        st.session_state.confirm_delete = row["id"]
                        st.warning(f"Are you sure you want to delete cluster '{row['name']}'? Click Delete again to confirm.")
            
            st.divider()
    
    # Register new cluster
    st.subheader("Register New Cluster")
    
    with st.form("register_cluster"):
        name = st.text_input("Cluster Name", key="cluster_name")
        api_server = st.text_input("API Server URL", key="api_server")
        description = st.text_area("Description", key="description")
        
        # Authentication
        auth_method = st.radio(
            "Authentication Method",
            ["Token", "Certificate", "Config File"],
            key="auth_method",
        )
        
        if auth_method == "Token":
            token = st.text_input("Authentication Token", type="password", key="token")
            certificate = None
            config_file = None
            config_context = None
        elif auth_method == "Certificate":
            token = None
            certificate = st.text_area("CA Certificate", key="certificate")
            config_file = None
            config_context = None
        else:  # Config File
            token = None
            certificate = None
            config_file = st.text_input("Config File Path", key="config_file")
            config_context = st.text_input("Config Context (optional)", key="config_context")
        
        insecure_skip_tls_verify = st.checkbox("Skip TLS Verification", key="insecure_skip_tls_verify")
        
        submitted = st.form_submit_button("Register Cluster")
        
        if submitted:
            if not name or not api_server:
                st.error("Cluster name and API server URL are required.")
                return
            
            # Create cluster request
            cluster_data = {
                "name": name,
                "api_server": api_server,
                "description": description,
                "insecure_skip_tls_verify": insecure_skip_tls_verify,
            }
            
            if auth_method == "Token" and token:
                cluster_data["token"] = token
            elif auth_method == "Certificate" and certificate:
                cluster_data["certificate"] = certificate
            elif auth_method == "Config File" and config_file:
                cluster_data["config_file"] = config_file
                if config_context:
                    cluster_data["config_context"] = config_context
            
            # Register cluster
            result = run_async(api_request("post", "/clusters", cluster_data))
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Cluster '{result['name']}' registered successfully!")
                st.rerun()


def cluster_details_page():
    """Cluster details page."""
    if "selected_cluster" not in st.session_state:
        st.error("No cluster selected.")
        return
    
    cluster_id = st.session_state.selected_cluster
    
    # Get cluster details
    cluster = run_async(api_request("get", f"/clusters/{cluster_id}"))
    
    if "error" in cluster:
        st.error(cluster["error"])
        return
    
    # Display cluster details
    st.title(f"Cluster: {cluster['name']}")
    
    # Status indicator
    status = cluster["health_status"] or "Unknown"
    if status == "healthy":
        st.success(f"Status: {status}")
    elif status == "warning":
        st.warning(f"Status: {status}")
    elif status == "critical":
        st.error(f"Status: {status}")
    else:
        st.info(f"Status: {status}")
    
    # Cluster information
    st.subheader("Cluster Information")
    st.markdown(f"**API Server:** {cluster['api_server']}")
    if cluster["description"]:
        st.markdown(f"**Description:** {cluster['description']}")
    st.markdown(f"**Created:** {cluster['created_at']}")
    st.markdown(f"**Updated:** {cluster['updated_at']}")
    
    # Actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Check Health"):
            st.session_state.page = "cluster_health"
            st.rerun()
    
    with col2:
        if st.button("Back to Clusters"):
            st.session_state.page = "clusters"
            st.rerun()
    
    with col3:
        if st.button("Delete Cluster"):
            if st.session_state.get("confirm_delete") == cluster_id:
                # Confirmed delete
                result = run_async(api_request("delete", f"/clusters/{cluster_id}"))
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(result["message"])
                    st.session_state.pop("confirm_delete", None)
                    st.session_state.page = "clusters"
                    st.rerun()
            else:
                # Ask for confirmation
                st.session_state.confirm_delete = cluster_id
                st.warning(f"Are you sure you want to delete cluster '{cluster['name']}'? Click Delete Cluster again to confirm.")


def cluster_health_page():
    """Cluster health page."""
    if "selected_cluster" not in st.session_state:
        st.error("No cluster selected.")
        return
    
    cluster_id = st.session_state.selected_cluster
    
    # Get cluster details
    cluster = run_async(api_request("get", f"/clusters/{cluster_id}"))
    
    if "error" in cluster:
        st.error(cluster["error"])
        return
    
    st.title(f"Health Check: {cluster['name']}")
    
    # Get health check
    with st.spinner("Checking cluster health..."):
        health_check = run_async(api_request("get", f"/clusters/{cluster_id}/health"))
    
    if "error" in health_check:
        st.error(health_check["error"])
        return
    
    # Display health status
    status = health_check["status"]
    if status == "healthy":
        st.success(f"Status: {status}")
    elif status == "warning":
        st.warning(f"Status: {status}")
    elif status == "critical":
        st.error(f"Status: {status}")
    else:
        st.info(f"Status: {status}")
    
    # Display issues
    if health_check["issues"]:
        st.subheader("Issues")
        
        for issue in health_check["issues"]:
            severity = issue.get("severity", "unknown")
            component = issue.get("component", "unknown")
            message = issue.get("message", "No message")
            
            if severity == "critical":
                st.error(f"**{component}:** {message}")
            elif severity == "warning":
                st.warning(f"**{component}:** {message}")
            else:
                st.info(f"**{component}:** {message}")
    
    # Display pod status
    st.subheader("Pod Status")
    
    pod_status = health_check["pod_status"]
    if "error" in pod_status:
        st.error(pod_status["error"])
    else:
        # Create pie chart for pod status
        status_count = pod_status.get("status_count", {})
        if status_count:
            labels = list(status_count.keys())
            values = list(status_count.values())
            
            fig = px.pie(
                names=labels,
                values=values,
                title="Pod Status Distribution",
                color=labels,
                color_discrete_map={
                    "Running": "#4CAF50",
                    "Pending": "#FFC107",
                    "Succeeded": "#2196F3",
                    "Failed": "#F44336",
                    "Unknown": "#9E9E9E",
                },
            )
            st.plotly_chart(fig)
        
        st.markdown(f"**Total Pods:** {pod_status.get('total', 0)}")
    
    # Display resource usage
    st.subheader("Resource Usage")
    
    resource_usage = health_check["resource_usage"]
    if "error" in resource_usage:
        st.error(resource_usage["error"])
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU usage
            cpu = resource_usage.get("cpu", {})
            if cpu:
                cpu_percent = cpu.get("usage_percent", 0)
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=cpu_percent,
                    title={"text": "CPU Usage"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#1E88E5"},
                        "steps": [
                            {"range": [0, 70], "color": "#E8F5E9"},
                            {"range": [70, 90], "color": "#FFF9C4"},
                            {"range": [90, 100], "color": "#FFEBEE"},
                        ],
                    },
                ))
                st.plotly_chart(fig)
        
        with col2:
            # Memory usage
            memory = resource_usage.get("memory", {})
            if memory:
                memory_percent = memory.get("usage_percent", 0)
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=memory_percent,
                    title={"text": "Memory Usage"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#43A047"},
                        "steps": [
                            {"range": [0, 70], "color": "#E8F5E9"},
                            {"range": [70, 90], "color": "#FFF9C4"},
                            {"range": [90, 100], "color": "#FFEBEE"},
                        ],
                    },
                ))
                st.plotly_chart(fig)
    
    # Display storage status
    st.subheader("Storage Status")
    
    storage_status = health_check["storage_status"]
    if "error" in storage_status:
        st.error(storage_status["error"])
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            # PVC status
            pvc_status = storage_status.get("pvc_status", {})
            if pvc_status:
                labels = list(pvc_status.keys())
                values = list(pvc_status.values())
                
                fig = px.pie(
                    names=labels,
                    values=values,
                    title="PVC Status Distribution",
                    color=labels,
                    color_discrete_map={
                        "Bound": "#4CAF50",
                        "Pending": "#FFC107",
                        "Lost": "#F44336",
                    },
                )
                st.plotly_chart(fig)
        
        with col2:
            # PV status
            pv_status = storage_status.get("pv_status", {})
            if pv_status:
                labels = list(pv_status.keys())
                values = list(pv_status.values())
                
                fig = px.pie(
                    names=labels,
                    values=values,
                    title="PV Status Distribution",
                    color=labels,
                    color_discrete_map={
                        "Bound": "#4CAF50",
                        "Available": "#2196F3",
                        "Released": "#FFC107",
                        "Failed": "#F44336",
                    },
                )
                st.plotly_chart(fig)
    
    # Back button
    if st.button("Back to Cluster Details"):
        st.session_state.page = "cluster_details"
        st.rerun()


def main():
    """Main function."""
    st.set_page_config(
        page_title="IronBox",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Sidebar
    st.sidebar.title("IronBox")
    st.sidebar.markdown("Multi-agent K8s management platform")
    
    # Navigation
    pages = {
        "chat": "Chat",
        "clusters": "Clusters",
        "cluster_details": "Cluster Details",
        "cluster_health": "Cluster Health",
    }
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "chat"
    
    # Show navigation only for main pages
    visible_pages = ["chat", "clusters"]
    for page in visible_pages:
        if st.sidebar.button(pages[page], key=f"nav_{page}"):
            st.session_state.page = page
            st.rerun()
    
    # Display current page
    current_page = st.session_state.page
    
    if current_page == "chat":
        chat_page()
    elif current_page == "clusters":
        clusters_page()
    elif current_page == "cluster_details":
        cluster_details_page()
    elif current_page == "cluster_health":
        cluster_health_page()
    else:
        st.error(f"Unknown page: {current_page}")


def run():
    """Run the Streamlit app."""
    main()


if __name__ == "__main__":
    run()
