"""
Test for LangChain-based agent frameworks with LangGraph.
"""
import os
import sys
import asyncio
import unittest
from typing import Dict, Any

import pytest

from ironbox.core.agent_core import AgentCore
from ironbox.core.langchain_frameworks import (
    BaseLCAgentFramework,
    LCRouteAgentFramework,
    LCReactAgentFramework,
    LCPlanAgentFramework,
    FrameworkRegistry
)
from ironbox.core.framework_selector import FrameworkSelector
from ironbox.core.toolkit import Toolkit
from ironbox.core.llm import default_llm
from ironbox.config import load_config


class TestLangChainFrameworks(unittest.TestCase):
    """Test LangChain-based frameworks."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.config = load_config()
        cls.agent_core = AgentCore(config=cls.config)
        asyncio.run(cls.agent_core.initialize())
    
    def test_framework_registry(self):
        """Test framework registry."""
        # Check if framework registry has the expected framework types
        self.assertIn("route", self.agent_core.framework_registry.framework_types)
        self.assertIn("react", self.agent_core.framework_registry.framework_types)
        self.assertIn("plan", self.agent_core.framework_registry.framework_types)
    
    def test_framework_selector(self):
        """Test framework selector."""
        # Check if framework selector is initialized
        self.assertIsNotNone(self.agent_core.framework_selector)
        
        # Test framework selection synchronously
        loop = asyncio.get_event_loop()
        
        # Simple question should select direct or route
        framework_type = loop.run_until_complete(
            self.agent_core.framework_selector.select_framework(
                "What is Kubernetes?"
            )
        )
        self.assertIn(framework_type, ["direct", "route"])
        
        # Complex reasoning should select react
        framework_type = loop.run_until_complete(
            self.agent_core.framework_selector.select_framework(
                "I need to analyze my cluster health and determine which pods are using the most memory."
            )
        )
        self.assertEqual(framework_type, "react")
        
        # Multi-step planning should select plan
        framework_type = loop.run_until_complete(
            self.agent_core.framework_selector.select_framework(
                "I need to migrate my application from one cluster to another. Can you help me plan the steps?"
            )
        )
        self.assertEqual(framework_type, "plan")
    
    @pytest.mark.asyncio
    async def test_process_query(self):
        """Test processing a query."""
        # Test a simple question
        result = await self.agent_core.process_query("What is Kubernetes?")
        
        # Verify that we got a response
        self.assertIn("response", result)
        self.assertNotEqual(result["response"], "")
        
        # Test a more complex query
        result = await self.agent_core.process_query(
            "I need to analyze my cluster health. Can you help me?"
        )
        
        # Verify that we got a response
        self.assertIn("response", result)
        self.assertNotEqual(result["response"], "")
        
        # Verify that we have agent outputs
        self.assertIn("agent_outputs", result)
        self.assertIsInstance(result["agent_outputs"], dict)


class TestFrameworkGraphFlow(unittest.TestCase):
    """Test the flow through the framework graph."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.config = load_config()
        cls.agent_core = AgentCore(config=cls.config)
        asyncio.run(cls.agent_core.initialize())
    
    @pytest.mark.asyncio
    async def test_graph_flow(self):
        """Test the flow through the graph."""
        # Test a query that should go through the graph
        result = await self.agent_core.process_query(
            "Can you help me troubleshoot my Kubernetes cluster?"
        )
        
        # Verify that we got a response
        self.assertIn("response", result)
        self.assertNotEqual(result["response"], "")
        
        # Verify that we have agent outputs
        self.assertIn("agent_outputs", result)
        self.assertIsInstance(result["agent_outputs"], dict)


if __name__ == "__main__":
    pytest.main(["-xvs", "test_langgraph_framework.py"])
