-- SQLite database initialization script for IronBox

-- Drop tables if they exist
DROP TABLE IF EXISTS function_calls;
DROP TABLE IF EXISTS chat_history;
DROP TABLE IF EXISTS cluster_health_checks;
DROP TABLE IF EXISTS clusters;
DROP TABLE IF EXISTS mcp_resources;
DROP TABLE IF EXISTS mcp_tools;
DROP TABLE IF EXISTS mcp_servers;

-- Create clusters table
CREATE TABLE clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    api_server VARCHAR(255) NOT NULL,
    config_context VARCHAR(100),
    config_file TEXT,
    token TEXT,
    certificate TEXT,
    insecure_skip_tls_verify BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_health_check TIMESTAMP,
    health_status VARCHAR(20),
    extra_data JSON
);

-- Create cluster_health_checks table
CREATE TABLE cluster_health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL,
    pod_status JSON,
    resource_usage JSON,
    storage_status JSON,
    issues JSON,
    details JSON,
    FOREIGN KEY (cluster_id) REFERENCES clusters (id) ON DELETE CASCADE
);

-- Create chat_history table
CREATE TABLE chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    extra_data JSON,
    -- Create index on session_id for faster lookups
    CONSTRAINT idx_chat_history_session_id UNIQUE (session_id, id)
);
CREATE INDEX idx_chat_history_session_id ON chat_history(session_id);

-- Create function_calls table
CREATE TABLE function_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    function_name VARCHAR(100) NOT NULL,
    parameters JSON,
    result JSON,
    -- Create index on session_id for faster lookups
    CONSTRAINT idx_function_calls_session_id UNIQUE (session_id, id)
);
CREATE INDEX idx_function_calls_session_id ON function_calls(session_id);

-- Create triggers to update the updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_clusters_timestamp
AFTER UPDATE ON clusters
BEGIN
    UPDATE clusters SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Create MCP servers table
CREATE TABLE mcp_servers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    version VARCHAR(50),
    command TEXT NOT NULL,
    args JSON,
    env JSON,
    enabled BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_connected_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'unknown',  -- "unknown", "connected", "disconnected", "error"
    error_message TEXT,
    capabilities JSON,
    extra_data JSON
);

-- Create MCP tools table
CREATE TABLE mcp_tools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id INTEGER NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    input_schema JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    extra_data JSON,
    FOREIGN KEY (server_id) REFERENCES mcp_servers (id) ON DELETE CASCADE,
    CONSTRAINT unique_tool_per_server UNIQUE (server_id, name)
);

-- Create MCP resources table
CREATE TABLE mcp_resources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id INTEGER NOT NULL,
    uri VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    description TEXT,
    mime_type VARCHAR(100),
    is_template BOOLEAN DEFAULT 0,
    uri_template VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    extra_data JSON,
    FOREIGN KEY (server_id) REFERENCES mcp_servers (id) ON DELETE CASCADE,
    CONSTRAINT unique_resource_per_server UNIQUE (server_id, uri)
);

-- Create triggers to update the updated_at timestamp for MCP tables
CREATE TRIGGER IF NOT EXISTS update_mcp_servers_timestamp
AFTER UPDATE ON mcp_servers
BEGIN
    UPDATE mcp_servers SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_mcp_tools_timestamp
AFTER UPDATE ON mcp_tools
BEGIN
    UPDATE mcp_tools SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_mcp_resources_timestamp
AFTER UPDATE ON mcp_resources
BEGIN
    UPDATE mcp_resources SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
