#!/usr/bin/env python3
"""
MCP Server for Qlib - Quantitative Investment Platform
Provides AI assistants with tools to interact with Qlib workflows.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import mcp.server
import mcp.server.stdio
import mcp.types as types

# Add qlib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qlib.config import C
from qlib.data import D
from qlib.workflow import R
from qlib.workflow.exp import Experiment
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.utils import init_instance_by_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QlibMCPServer:
    """MCP Server providing Qlib quantitative investment tools."""
    
    def __init__(self):
        self.server = mcp.server.Server("qlib")
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List all available tools."""
            return [
                types.Tool(
                    name="initialize_qlib",
                    description="Initialize Qlib with configuration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "provider_uri": {
                                "type": "string",
                                "description": "Data provider URI"
                            },
                            "region": {
                                "type": "string",
                                "description": "Region for data (e.g., 'cn', 'us')"
                            }
                        }
                    }
                ),
                types.Tool(
                    name="get_stock_data",
                    description="Get stock data for specified symbols and time range",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of stock symbols"
                            },
                            "start_time": {
                                "type": "string",
                                "description": "Start time (YYYY-MM-DD)"
                            },
                            "end_time": {
                                "type": "string",
                                "description": "End time (YYYY-MM-DD)"
                            },
                            "fields": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Data fields to retrieve"
                            }
                        },
                        "required": ["symbols", "start_time", "end_time"]
                    }
                ),
                types.Tool(
                    name="run_backtest",
                    description="Run backtest with specified strategy and data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "strategy": {
                                "type": "string",
                                "description": "Strategy name (e.g., 'TopkDropoutStrategy')"
                            },
                            "benchmark": {
                                "type": "string",
                                "description": "Benchmark symbol"
                            },
                            "start_time": {
                                "type": "string",
                                "description": "Start time (YYYY-MM-DD)"
                            },
                            "end_time": {
                                "type": "string",
                                "description": "End time (YYYY-MM-DD)"
                            },
                            "topk": {
                                "type": "integer",
                                "description": "Number of top stocks to select"
                            }
                        },
                        "required": ["strategy", "benchmark", "start_time", "end_time"]
                    }
                ),
                types.Tool(
                    name="train_model",
                    description="Train a machine learning model with specified configuration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Model name (e.g., 'LGBModel', 'MLPModel')"
                            },
                            "dataset": {
                                "type": "string",
                                "description": "Dataset name"
                            },
                            "feature_columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Feature column names"
                            },
                            "label_columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Label column names"
                            }
                        },
                        "required": ["model_name", "dataset"]
                    }
                ),
                types.Tool(
                    name="get_portfolio_analysis",
                    description="Get portfolio analysis and performance metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "portfolio_path": {
                                "type": "string",
                                "description": "Path to portfolio results"
                            }
                        }
                    }
                ),
                types.Tool(
                    name="list_available_models",
                    description="List all available machine learning models in Qlib",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="get_workflow_status",
                    description="Get current workflow execution status",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls."""
            try:
                if name == "initialize_qlib":
                    return await self._initialize_qlib(arguments)
                elif name == "get_stock_data":
                    return await self._get_stock_data(arguments)
                elif name == "run_backtest":
                    return await self._run_backtest(arguments)
                elif name == "train_model":
                    return await self._train_model(arguments)
                elif name == "get_portfolio_analysis":
                    return await self._get_portfolio_analysis(arguments)
                elif name == "list_available_models":
                    return await self._list_available_models(arguments)
                elif name == "get_workflow_status":
                    return await self._get_workflow_status(arguments)
                else:
                    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _initialize_qlib(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Initialize Qlib with configuration."""
        try:
            provider_uri = args.get("provider_uri", "~/.qlib/qlib_data/cn_data")
            region = args.get("region", "cn")
            
            C.set("provider_uri", provider_uri)
            C.set("region", region)
            
            # Initialize data provider
            D.init(provider_uri, region)
            
            return [types.TextContent(
                type="text",
                text=f"Qlib initialized successfully with provider: {provider_uri}, region: {region}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Failed to initialize Qlib: {str(e)}")]
    
    async def _get_stock_data(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Get stock data for specified symbols and time range."""
        try:
            symbols = args["symbols"]
            start_time = args["start_time"]
            end_time = args["end_time"]
            fields = args.get("fields", ["$close", "$volume", "$factor"])
            
            # Get data
            data = D.features(symbols, fields, start_time, end_time)
            
            # Convert to readable format
            result = f"Retrieved data for {len(symbols)} symbols from {start_time} to {end_time}\n"
            result += f"Data shape: {data.shape}\n"
            result += f"Fields: {', '.join(fields)}\n"
            result += f"Sample data:\n{data.head()}"
            
            return [types.TextContent(type="text", text=result)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Failed to get stock data: {str(e)}")]
    
    async def _run_backtest(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Run backtest with specified strategy and data."""
        try:
            strategy_name = args["strategy"]
            benchmark = args["benchmark"]
            start_time = args["start_time"]
            end_time = args["end_time"]
            topk = args.get("topk", 50)
            
            # Create strategy
            if strategy_name == "TopkDropoutStrategy":
                strategy = TopkDropoutStrategy(topk=topk)
            else:
                return [types.TextContent(type="text", text=f"Unknown strategy: {strategy_name}")]
            
            # Run backtest
            portfolio_config = {
                "benchmark": benchmark,
                "account": 100000000,
                "exchange_kwargs": {
                    "freq": "day",
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            }
            
            with R.start(experiment_name="backtest"):
                sr = SignalRecord(model=strategy, dataset="", port_analysis_config=portfolio_config)
                sr.generate()
            
            return [types.TextContent(
                type="text",
                text=f"Backtest completed successfully for {strategy_name} from {start_time} to {end_time}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Failed to run backtest: {str(e)}")]
    
    async def _train_model(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Train a machine learning model with specified configuration."""
        try:
            model_name = args["model_name"]
            dataset = args["dataset"]
            feature_columns = args.get("feature_columns", ["$close", "$volume", "$factor"])
            label_columns = args.get("label_columns", ["Ref($close, -1)/$close - 1"])
            
            # Create model configuration
            model_config = {
                "class": f"qlib.contrib.model.{model_name}",
                "module_path": f"qlib.contrib.model.{model_name}",
                "kwargs": {
                    "feature_columns": feature_columns,
                    "label_columns": label_columns,
                }
            }
            
            # Create dataset configuration
            dataset_config = {
                "class": "qlib.data.dataset.DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": "qlib.data.dataset.handler.Alpha158",
                        "module_path": "qlib.data.dataset.handler",
                        "kwargs": {
                            "start_time": "2008-01-01",
                            "end_time": "2020-12-31",
                            "fit_start_time": "2008-01-01",
                            "fit_end_time": "2014-12-31",
                            "instruments": "csi300",
                        }
                    }
                }
            }
            
            # Run training
            with R.start(experiment_name=f"train_{model_name}"):
                model = init_instance_by_config(model_config)
                dataset = init_instance_by_config(dataset_config)
                
                with R.start(record_name="train"):
                    model.fit(dataset)
                    R.save_objects(model=model)
            
            return [types.TextContent(
                type="text",
                text=f"Model {model_name} trained successfully on dataset {dataset}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Failed to train model: {str(e)}")]
    
    async def _get_portfolio_analysis(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Get portfolio analysis and performance metrics."""
        try:
            portfolio_path = args.get("portfolio_path", "")
            
            if not portfolio_path:
                # Get latest portfolio
                portfolios = R.list_records()
                if not portfolios:
                    return [types.TextContent(type="text", text="No portfolio records found")]
                portfolio_path = portfolios[-1]
            
            # Load portfolio analysis
            with R.start(experiment_name=portfolio_path):
                analysis = PortAnaRecord.load("analysis")
                report = analysis.generate()
            
            return [types.TextContent(
                type="text",
                text=f"Portfolio analysis for {portfolio_path}:\n{report}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Failed to get portfolio analysis: {str(e)}")]
    
    async def _list_available_models(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """List all available machine learning models in Qlib."""
        try:
            models = [
                "LGBModel", "MLPModel", "GRUModel", "LSTMModel", "TransformerModel",
                "TFTModel", "TabNetModel", "CatBoostModel", "XGBoostModel",
                "LinearModel", "EnsembleModel", "MetaModel"
            ]
            
            result = "Available models in Qlib:\n"
            for model in models:
                result += f"- {model}\n"
            
            return [types.TextContent(type="text", text=result)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Failed to list models: {str(e)}")]
    
    async def _get_workflow_status(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Get current workflow execution status."""
        try:
            experiments = R.list_experiments()
            records = R.list_records()
            
            result = f"Workflow Status:\n"
            result += f"Active experiments: {len(experiments)}\n"
            result += f"Total records: {len(records)}\n"
            
            if experiments:
                result += f"Latest experiment: {experiments[-1]}\n"
            if records:
                result += f"Latest record: {records[-1]}\n"
            
            return [types.TextContent(type="text", text=result)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Failed to get workflow status: {str(e)}")]
    
    async def run(self):
        """Run the MCP server."""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                server_logger=logger
            )

async def main():
    """Main entry point."""
    server = QlibMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main()) 