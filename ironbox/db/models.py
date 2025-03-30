"""
Database models for IronBox.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Cluster(Base):
    """
    Model for Kubernetes cluster information.
    """
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    api_server = Column(String(255), nullable=False)
    config_context = Column(String(100), nullable=True)
    config_file = Column(Text, nullable=True)
    token = Column(Text, nullable=True)
    certificate = Column(Text, nullable=True)
    insecure_skip_tls_verify = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_health_check = Column(DateTime, nullable=True)
    health_status = Column(String(20), nullable=True)  # "healthy", "warning", "critical"
    extra_data = Column(JSON, nullable=True)  # Additional metadata
    
    # Relationships
    health_checks = relationship("ClusterHealthCheck", back_populates="cluster", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Cluster(name='{self.name}', api_server='{self.api_server}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "api_server": self.api_server,
            "config_context": self.config_context,
            "insecure_skip_tls_verify": self.insecure_skip_tls_verify,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "health_status": self.health_status,
            "extra_data": self.extra_data,
        }


class ClusterHealthCheck(Base):
    """
    Model for cluster health check results.
    """
    __tablename__ = "cluster_health_checks"

    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey("clusters.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), nullable=False)  # "healthy", "warning", "critical"
    pod_status = Column(JSON, nullable=True)  # Summary of pod statuses
    resource_usage = Column(JSON, nullable=True)  # CPU, memory usage
    storage_status = Column(JSON, nullable=True)  # PVC/Volume health
    issues = Column(JSON, nullable=True)  # List of detected issues
    details = Column(JSON, nullable=True)  # Detailed check results
    
    # Relationships
    cluster = relationship("Cluster", back_populates="health_checks")
    
    def __repr__(self) -> str:
        return f"<ClusterHealthCheck(cluster_id={self.cluster_id}, status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "cluster_id": self.cluster_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "status": self.status,
            "pod_status": self.pod_status,
            "resource_usage": self.resource_usage,
            "storage_status": self.storage_status,
            "issues": self.issues,
            "details": self.details,
        }


class ChatHistory(Base):
    """
    Model for storing chat history.
    """
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    role = Column(String(20), nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)
    extra_data = Column(JSON, nullable=True)  # Additional metadata
    
    def __repr__(self) -> str:
        return f"<ChatHistory(session_id='{self.session_id}', role='{self.role}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "role": self.role,
            "content": self.content,
            "extra_data": self.extra_data,
        }


class FunctionCall(Base):
    """
    Model for storing function call parameters.
    """
    __tablename__ = "function_calls"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    function_name = Column(String(100), nullable=False)
    parameters = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    
    def __repr__(self) -> str:
        return f"<FunctionCall(session_id='{self.session_id}', function_name='{self.function_name}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "function_name": self.function_name,
            "parameters": self.parameters,
            "result": self.result,
        }


class MCPServer(Base):
    """
    Model for MCP server information.
    """
    __tablename__ = "mcp_servers"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    version = Column(String(50), nullable=True)
    command = Column(Text, nullable=False)
    args = Column(JSON, nullable=True)
    env = Column(JSON, nullable=True)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_connected_at = Column(DateTime, nullable=True)
    status = Column(String(20), default="unknown")  # "unknown", "connected", "disconnected", "error"
    error_message = Column(Text, nullable=True)
    capabilities = Column(JSON, nullable=True)
    extra_data = Column(JSON, nullable=True)
    
    # Relationships
    tools = relationship("MCPTool", back_populates="server", cascade="all, delete-orphan")
    resources = relationship("MCPResource", back_populates="server", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<MCPServer(name='{self.name}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_connected_at": self.last_connected_at.isoformat() if self.last_connected_at else None,
            "status": self.status,
            "error_message": self.error_message,
            "capabilities": self.capabilities,
            "extra_data": self.extra_data,
        }


class MCPTool(Base):
    """
    Model for MCP tool information.
    """
    __tablename__ = "mcp_tools"

    id = Column(Integer, primary_key=True)
    server_id = Column(Integer, ForeignKey("mcp_servers.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    input_schema = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    extra_data = Column(JSON, nullable=True)
    
    # Relationships
    server = relationship("MCPServer", back_populates="tools")
    
    def __repr__(self) -> str:
        return f"<MCPTool(name='{self.name}', server_id={self.server_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "server_id": self.server_id,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "extra_data": self.extra_data,
        }


class MCPResource(Base):
    """
    Model for MCP resource information.
    """
    __tablename__ = "mcp_resources"

    id = Column(Integer, primary_key=True)
    server_id = Column(Integer, ForeignKey("mcp_servers.id"), nullable=False)
    uri = Column(String(255), nullable=False)
    name = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    mime_type = Column(String(100), nullable=True)
    is_template = Column(Boolean, default=False)
    uri_template = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    extra_data = Column(JSON, nullable=True)
    
    # Relationships
    server = relationship("MCPServer", back_populates="resources")
    
    def __repr__(self) -> str:
        return f"<MCPResource(uri='{self.uri}', server_id={self.server_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "server_id": self.server_id,
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mime_type": self.mime_type,
            "is_template": self.is_template,
            "uri_template": self.uri_template,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "extra_data": self.extra_data,
        }
