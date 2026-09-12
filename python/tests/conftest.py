"""
测试配置 & 共享 fixture

前提：运行测试前需配置 .env 文件（含 OPENAI_API_KEY），
     或设置环境变量 OPENAI_API_KEY=your-key。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from dotenv import load_dotenv

# 锚定项目根目录加载 .env——与 pytest 的运行 cwd 无关。
# 否则从仓库根目录跑测试时 pydantic-settings 找不到 .env，
# ChatOpenAI 初始化因缺 API key 直接 import 失败。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@pytest.fixture
def mock_vector_store():
    """Mock 向量库服务"""
    store = MagicMock()
    store.search = AsyncMock(return_value=[
        ({"content": "test doc content", "source": "test.pdf", "metadata": {}}, 0.95),
        ({"content": "another doc", "source": "doc2.pdf", "metadata": {}}, 0.80),
    ])
    store.get_stats = AsyncMock(return_value={"backend": "chroma", "total_vectors": 42})
    store.add_chunks = AsyncMock(return_value=5)
    store.init = AsyncMock()
    return store


@pytest.fixture
def mock_knowledge_graph():
    """Mock 知识图谱服务"""
    kg = MagicMock()
    kg.execute_cypher = AsyncMock(return_value=[
        {"n": {"name": "张三", "type": "Person"}},
        {"r": "works_at"},
    ])
    kg.get_neighbors = AsyncMock(return_value=[
        {"entity": "张三", "relation": "works_at", "target": "腾讯"},
    ])
    kg.get_stats = AsyncMock(return_value={"entities": 100, "relations": 250})
    kg.init = AsyncMock()
    kg.close = AsyncMock()
    return kg


@pytest.fixture
def sample_messages():
    """模拟 ReAct agent 返回的 messages 列表"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    return [
        HumanMessage(content="张三是谁？"),
        AIMessage(
            content="",
            tool_calls=[{"name": "entity_lookup", "args": {"entity_name": "张三"}, "id": "call_1"}],
        ),
        ToolMessage(
            content=(
                '{"entity":"张三","relationships":[{"entity":"张三",'
                '"relation":"works_at","target":"腾讯"}],"contexts":['
                '{"content":"张三 works_at 腾讯","source":"knowledge_graph",'
                '"score":0.8,"retrieval_type":"graph"}]}'
            ),
            name="entity_lookup",
            tool_call_id="call_1",
        ),
        AIMessage(content="根据知识图谱查询结果，张三是腾讯的员工。"),
    ]


@pytest.fixture
def sample_text():
    """用于测试分块的文本"""
    return (
        "LangGraph 是一个用于构建有状态、多角色 LLM 应用的框架。"
        "它基于有向图的概念，将 Agent 的执行流程建模为节点和边的组合。"
        "每个节点代表一个处理步骤，边定义了节点之间的数据流向。"
        "StateGraph 是 LangGraph 的核心抽象，它维护了一个在节点间传递的共享状态。"
        "通过 checkpoint 机制，LangGraph 支持状态的持久化和恢复。"
        "这使得 Agent 可以暂停执行、等待人工审批、然后继续运行。"
        * 10  # 重复 10 次让文本足够长
    )
