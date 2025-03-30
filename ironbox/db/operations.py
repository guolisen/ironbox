"""
Database operations for IronBox.
"""
import logging
from typing import List, Dict, Any, Optional, Union, Type, TypeVar

from sqlalchemy import create_engine, select, update, delete
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select as future_select

from ironbox.config import config
from ironbox.db.models import Base, Cluster, ClusterHealthCheck, ChatHistory, FunctionCall, MCPServer, MCPTool, MCPResource

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for models
T = TypeVar('T', bound=Base)

# Create engine and session
engine = create_async_engine(
    config["database"]["url"].replace("sqlite://", "sqlite+aiosqlite://"),
    echo=config["database"]["echo"],
)
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db():
    """Initialize the database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")


async def get_db_session():
    """Get a database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


class DatabaseOperations:
    """Database operations for IronBox."""

    @staticmethod
    async def create_item(session: AsyncSession, model: Type[T], data: Dict[str, Any]) -> T:
        """
        Create a new item in the database.
        
        Args:
            session: Database session
            model: Model class
            data: Item data
            
        Returns:
            Created item
        """
        item = model(**data)
        session.add(item)
        await session.commit()
        await session.refresh(item)
        return item

    @staticmethod
    async def get_item(session: AsyncSession, model: Type[T], item_id: int) -> Optional[T]:
        """
        Get an item by ID.
        
        Args:
            session: Database session
            model: Model class
            item_id: Item ID
            
        Returns:
            Item or None if not found
        """
        result = await session.execute(future_select(model).filter(model.id == item_id))
        return result.scalars().first()

    @staticmethod
    async def get_items(session: AsyncSession, model: Type[T], filters: Optional[Dict[str, Any]] = None) -> List[T]:
        """
        Get items with optional filters.
        
        Args:
            session: Database session
            model: Model class
            filters: Optional filters
            
        Returns:
            List of items
        """
        query = future_select(model)
        if filters:
            for key, value in filters.items():
                if hasattr(model, key):
                    query = query.filter(getattr(model, key) == value)
        result = await session.execute(query)
        return result.scalars().all()

    @staticmethod
    async def update_item(session: AsyncSession, model: Type[T], item_id: int, data: Dict[str, Any]) -> Optional[T]:
        """
        Update an item.
        
        Args:
            session: Database session
            model: Model class
            item_id: Item ID
            data: Updated data
            
        Returns:
            Updated item or None if not found
        """
        await session.execute(
            update(model).where(model.id == item_id).values(**data)
        )
        await session.commit()
        return await DatabaseOperations.get_item(session, model, item_id)

    @staticmethod
    async def delete_item(session: AsyncSession, model: Type[T], item_id: int) -> bool:
        """
        Delete an item.
        
        Args:
            session: Database session
            model: Model class
            item_id: Item ID
            
        Returns:
            True if deleted, False if not found
        """
        result = await session.execute(
            delete(model).where(model.id == item_id)
        )
        await session.commit()
        return result.rowcount > 0


class ClusterOperations:
    """Operations for Kubernetes clusters."""

    @staticmethod
    async def create_cluster(session: AsyncSession, data: Dict[str, Any]) -> Cluster:
        """
        Create a new cluster.
        
        Args:
            session: Database session
            data: Cluster data
            
        Returns:
            Created cluster
        """
        return await DatabaseOperations.create_item(session, Cluster, data)

    @staticmethod
    async def get_cluster(session: AsyncSession, cluster_id: int) -> Optional[Cluster]:
        """
        Get a cluster by ID.
        
        Args:
            session: Database session
            cluster_id: Cluster ID
            
        Returns:
            Cluster or None if not found
        """
        return await DatabaseOperations.get_item(session, Cluster, cluster_id)

    @staticmethod
    async def get_cluster_by_name(session: AsyncSession, name: str) -> Optional[Cluster]:
        """
        Get a cluster by name.
        
        Args:
            session: Database session
            name: Cluster name
            
        Returns:
            Cluster or None if not found
        """
        result = await session.execute(
            future_select(Cluster).filter(Cluster.name == name)
        )
        return result.scalars().first()

    @staticmethod
    async def get_clusters(session: AsyncSession) -> List[Cluster]:
        """
        Get all clusters.
        
        Args:
            session: Database session
            
        Returns:
            List of clusters
        """
        return await DatabaseOperations.get_items(session, Cluster)

    @staticmethod
    async def update_cluster(session: AsyncSession, cluster_id: int, data: Dict[str, Any]) -> Optional[Cluster]:
        """
        Update a cluster.
        
        Args:
            session: Database session
            cluster_id: Cluster ID
            data: Updated data
            
        Returns:
            Updated cluster or None if not found
        """
        return await DatabaseOperations.update_item(session, Cluster, cluster_id, data)

    @staticmethod
    async def delete_cluster(session: AsyncSession, cluster_id: int) -> bool:
        """
        Delete a cluster.
        
        Args:
            session: Database session
            cluster_id: Cluster ID
            
        Returns:
            True if deleted, False if not found
        """
        return await DatabaseOperations.delete_item(session, Cluster, cluster_id)

    @staticmethod
    async def add_health_check(session: AsyncSession, cluster_id: int, health_data: Dict[str, Any]) -> ClusterHealthCheck:
        """
        Add a health check for a cluster.
        
        Args:
            session: Database session
            cluster_id: Cluster ID
            health_data: Health check data
            
        Returns:
            Created health check
        """
        health_data["cluster_id"] = cluster_id
        health_check = await DatabaseOperations.create_item(session, ClusterHealthCheck, health_data)
        
        # Update cluster's last health check and status
        await ClusterOperations.update_cluster(
            session, 
            cluster_id, 
            {
                "last_health_check": health_check.timestamp,
                "health_status": health_check.status
            }
        )
        
        return health_check

    @staticmethod
    async def get_health_checks(session: AsyncSession, cluster_id: int, limit: int = 10) -> List[ClusterHealthCheck]:
        """
        Get health checks for a cluster.
        
        Args:
            session: Database session
            cluster_id: Cluster ID
            limit: Maximum number of health checks to return
            
        Returns:
            List of health checks
        """
        result = await session.execute(
            future_select(ClusterHealthCheck)
            .filter(ClusterHealthCheck.cluster_id == cluster_id)
            .order_by(ClusterHealthCheck.timestamp.desc())
            .limit(limit)
        )
        return result.scalars().all()


class MemoryOperations:
    """Operations for chat history and function calls."""

    @staticmethod
    async def add_chat_message(session: AsyncSession, session_id: str, role: str, content: str, extra_data: Optional[Dict[str, Any]] = None) -> ChatHistory:
        """
        Add a chat message to history.
        
        Args:
            session: Database session
            session_id: Chat session ID
            role: Message role (user, assistant, system)
            content: Message content
            extra_data: Optional extra data
            
        Returns:
            Created chat history entry
        """
        data = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "extra_data": extra_data or {},
        }
        return await DatabaseOperations.create_item(session, ChatHistory, data)

    @staticmethod
    async def get_chat_history(session: AsyncSession, session_id: str, limit: int = 50) -> List[ChatHistory]:
        """
        Get chat history for a session.
        
        Args:
            session: Database session
            session_id: Chat session ID
            limit: Maximum number of messages to return
            
        Returns:
            List of chat history entries
        """
        result = await session.execute(
            future_select(ChatHistory)
            .filter(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.timestamp.asc())
            .limit(limit)
        )
        return result.scalars().all()

    @staticmethod
    async def add_function_call(session: AsyncSession, session_id: str, function_name: str, parameters: Optional[Dict[str, Any]] = None, result: Optional[Dict[str, Any]] = None) -> FunctionCall:
        """
        Add a function call to history.
        
        Args:
            session: Database session
            session_id: Chat session ID
            function_name: Function name
            parameters: Function parameters
            result: Function result
            
        Returns:
            Created function call entry
        """
        data = {
            "session_id": session_id,
            "function_name": function_name,
            "parameters": parameters or {},
            "result": result or {},
        }
        return await DatabaseOperations.create_item(session, FunctionCall, data)

    @staticmethod
    async def get_function_calls(session: AsyncSession, session_id: str, function_name: Optional[str] = None, limit: int = 10) -> List[FunctionCall]:
        """
        Get function calls for a session.
        
        Args:
            session: Database session
            session_id: Chat session ID
            function_name: Optional function name filter
            limit: Maximum number of function calls to return
            
        Returns:
            List of function call entries
        """
        query = future_select(FunctionCall).filter(FunctionCall.session_id == session_id)
        if function_name:
            query = query.filter(FunctionCall.function_name == function_name)
        query = query.order_by(FunctionCall.timestamp.desc()).limit(limit)
        result = await session.execute(query)
        return result.scalars().all()


class MCPOperations:
    """Operations for MCP servers, tools, and resources."""

    @staticmethod
    async def create_server(session: AsyncSession, data: Dict[str, Any]) -> MCPServer:
        """
        Create a new MCP server.
        
        Args:
            session: Database session
            data: Server data
            
        Returns:
            Created server
        """
        return await DatabaseOperations.create_item(session, MCPServer, data)

    @staticmethod
    async def get_server(session: AsyncSession, server_id: int) -> Optional[MCPServer]:
        """
        Get an MCP server by ID.
        
        Args:
            session: Database session
            server_id: Server ID
            
        Returns:
            Server or None if not found
        """
        return await DatabaseOperations.get_item(session, MCPServer, server_id)

    @staticmethod
    async def get_server_by_name(session: AsyncSession, name: str) -> Optional[MCPServer]:
        """
        Get an MCP server by name.
        
        Args:
            session: Database session
            name: Server name
            
        Returns:
            Server or None if not found
        """
        result = await session.execute(
            future_select(MCPServer).filter(MCPServer.name == name)
        )
        return result.scalars().first()

    @staticmethod
    async def get_servers(session: AsyncSession, enabled_only: bool = False) -> List[MCPServer]:
        """
        Get all MCP servers.
        
        Args:
            session: Database session
            enabled_only: Whether to return only enabled servers
            
        Returns:
            List of servers
        """
        query = future_select(MCPServer)
        if enabled_only:
            query = query.filter(MCPServer.enabled == True)
        result = await session.execute(query)
        return result.scalars().all()

    @staticmethod
    async def update_server(session: AsyncSession, server_id: int, data: Dict[str, Any]) -> Optional[MCPServer]:
        """
        Update an MCP server.
        
        Args:
            session: Database session
            server_id: Server ID
            data: Updated data
            
        Returns:
            Updated server or None if not found
        """
        return await DatabaseOperations.update_item(session, MCPServer, server_id, data)

    @staticmethod
    async def delete_server(session: AsyncSession, server_id: int) -> bool:
        """
        Delete an MCP server.
        
        Args:
            session: Database session
            server_id: Server ID
            
        Returns:
            True if deleted, False if not found
        """
        return await DatabaseOperations.delete_item(session, MCPServer, server_id)

    @staticmethod
    async def create_tool(session: AsyncSession, server_id: int, data: Dict[str, Any]) -> MCPTool:
        """
        Create a new MCP tool.
        
        Args:
            session: Database session
            server_id: Server ID
            data: Tool data
            
        Returns:
            Created tool
        """
        data["server_id"] = server_id
        return await DatabaseOperations.create_item(session, MCPTool, data)

    @staticmethod
    async def get_tool(session: AsyncSession, tool_id: int) -> Optional[MCPTool]:
        """
        Get an MCP tool by ID.
        
        Args:
            session: Database session
            tool_id: Tool ID
            
        Returns:
            Tool or None if not found
        """
        return await DatabaseOperations.get_item(session, MCPTool, tool_id)

    @staticmethod
    async def get_tool_by_name(session: AsyncSession, server_id: int, name: str) -> Optional[MCPTool]:
        """
        Get an MCP tool by name.
        
        Args:
            session: Database session
            server_id: Server ID
            name: Tool name
            
        Returns:
            Tool or None if not found
        """
        result = await session.execute(
            future_select(MCPTool)
            .filter(MCPTool.server_id == server_id)
            .filter(MCPTool.name == name)
        )
        return result.scalars().first()

    @staticmethod
    async def get_tools(session: AsyncSession, server_id: int) -> List[MCPTool]:
        """
        Get all MCP tools for a server.
        
        Args:
            session: Database session
            server_id: Server ID
            
        Returns:
            List of tools
        """
        result = await session.execute(
            future_select(MCPTool).filter(MCPTool.server_id == server_id)
        )
        return result.scalars().all()

    @staticmethod
    async def update_tool(session: AsyncSession, tool_id: int, data: Dict[str, Any]) -> Optional[MCPTool]:
        """
        Update an MCP tool.
        
        Args:
            session: Database session
            tool_id: Tool ID
            data: Updated data
            
        Returns:
            Updated tool or None if not found
        """
        return await DatabaseOperations.update_item(session, MCPTool, tool_id, data)

    @staticmethod
    async def delete_tool(session: AsyncSession, tool_id: int) -> bool:
        """
        Delete an MCP tool.
        
        Args:
            session: Database session
            tool_id: Tool ID
            
        Returns:
            True if deleted, False if not found
        """
        return await DatabaseOperations.delete_item(session, MCPTool, tool_id)

    @staticmethod
    async def create_resource(session: AsyncSession, server_id: int, data: Dict[str, Any]) -> MCPResource:
        """
        Create a new MCP resource.
        
        Args:
            session: Database session
            server_id: Server ID
            data: Resource data
            
        Returns:
            Created resource
        """
        data["server_id"] = server_id
        return await DatabaseOperations.create_item(session, MCPResource, data)

    @staticmethod
    async def get_resource(session: AsyncSession, resource_id: int) -> Optional[MCPResource]:
        """
        Get an MCP resource by ID.
        
        Args:
            session: Database session
            resource_id: Resource ID
            
        Returns:
            Resource or None if not found
        """
        return await DatabaseOperations.get_item(session, MCPResource, resource_id)

    @staticmethod
    async def get_resource_by_uri(session: AsyncSession, server_id: int, uri: str) -> Optional[MCPResource]:
        """
        Get an MCP resource by URI.
        
        Args:
            session: Database session
            server_id: Server ID
            uri: Resource URI
            
        Returns:
            Resource or None if not found
        """
        result = await session.execute(
            future_select(MCPResource)
            .filter(MCPResource.server_id == server_id)
            .filter(MCPResource.uri == uri)
        )
        return result.scalars().first()

    @staticmethod
    async def get_resources(session: AsyncSession, server_id: int, is_template: Optional[bool] = None) -> List[MCPResource]:
        """
        Get all MCP resources for a server.
        
        Args:
            session: Database session
            server_id: Server ID
            is_template: Optional filter for resource templates
            
        Returns:
            List of resources
        """
        query = future_select(MCPResource).filter(MCPResource.server_id == server_id)
        if is_template is not None:
            query = query.filter(MCPResource.is_template == is_template)
        result = await session.execute(query)
        return result.scalars().all()

    @staticmethod
    async def update_resource(session: AsyncSession, resource_id: int, data: Dict[str, Any]) -> Optional[MCPResource]:
        """
        Update an MCP resource.
        
        Args:
            session: Database session
            resource_id: Resource ID
            data: Updated data
            
        Returns:
            Updated resource or None if not found
        """
        return await DatabaseOperations.update_item(session, MCPResource, resource_id, data)

    @staticmethod
    async def delete_resource(session: AsyncSession, resource_id: int) -> bool:
        """
        Delete an MCP resource.
        
        Args:
            session: Database session
            resource_id: Resource ID
            
        Returns:
            True if deleted, False if not found
        """
        return await DatabaseOperations.delete_item(session, MCPResource, resource_id)
