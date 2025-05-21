from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

# Initialize SQLite memory for state persistence
memory = SqliteSaver.from_conn_string(":memory:")

# Define the state structure for our legal research agent
class AgentState(TypedDict):
    legal_question: str
    research_plan: str
    sources: List[str]
    case_summaries: List[str]
    argument_draft: str
    critique: str
    revision_number: int
    max_revisions: int

from langchain_openai import ChatOpenAI
# Initialize the language model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# System prompts for different stages of legal research
RESEARCH_PLAN_PROMPT = """You are an experienced law librarian. Given the student's legal question, create a detailed research plan that includes:
1. Key legal concepts to research
2. Types of sources to consult (cases, statutes, regulations, secondary sources)
3. Suggested search strategies
4. Important jurisdictions to consider
5. Potential counterarguments to research

Provide a structured outline that will guide the research process."""

CASE_FINDER_PROMPT = """You are a legal research assistant. Based on the research plan and legal question, generate 3-5 specific search queries to find relevant:
- Case law
- Statutes
- Regulations
- Law review articles
- Secondary sources

Format each query to maximize relevant results."""

CASE_SUMMARIZER_PROMPT = """You are a law student summarizing legal sources. For each source provided, create a concise summary that includes:
1. Key facts
2. Legal issue(s)
3. Holding/rule
4. Reasoning
5. Important quotes
6. Citation

Focus on elements most relevant to the original legal question."""

ARGUMENT_BUILDER_PROMPT = """You are a law student writing a legal memorandum. Using the research plan, sources, and case summaries, draft a comprehensive legal argument that:
1. States the legal question
2. Presents relevant rules and precedents
3. Applies the law to the facts
4. Addresses potential counterarguments
5. Reaches a conclusion
6. Includes proper citations

Ensure your argument is clear, logical, and well-supported by authority."""

CRITIQUE_PROMPT = """You are a law professor reviewing a legal memorandum. Provide detailed feedback on:
1. Legal analysis and reasoning
2. Use of authority and citations
3. Structure and organization
4. Clarity and precision
5. Counterargument analysis
6. Areas for improvement

Be specific and constructive in your critique."""

from langchain_core.pydantic_v1 import BaseModel

# Define the structure for search queries
class Queries(BaseModel):
    queries: List[str]

from tavily import TavilyClient
import os

# Initialize Tavily client for web search
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Node functions for the state graph
def research_plan_node(state: AgentState):
    """Generate a research plan for the legal question."""
    messages = [
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['legal_question'])
    ]
    response = model.invoke(messages)
    return {"research_plan": response.content}

def case_finder_node(state: AgentState):
    """Generate search queries for finding legal sources."""
    messages = [
        SystemMessage(content=CASE_FINDER_PROMPT),
        HumanMessage(content=f"Question: {state['legal_question']}\nPlan: {state['research_plan']}")
    ]
    queries = model.with_structured_output(Queries).invoke(messages)
    sources = state['sources'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            sources.append(r['content'])
    return {"sources": sources}

def case_summarizer_node(state: AgentState):
    """Summarize each legal source."""
    summaries = []
    for source in state['sources']:
        messages = [
            SystemMessage(content=CASE_SUMMARIZER_PROMPT),
            HumanMessage(content=source)
        ]
        response = model.invoke(messages)
        summaries.append(response.content)
    return {"case_summaries": summaries}

def argument_builder_node(state: AgentState):
    """Draft a legal argument using the research and summaries."""
    content = "\n\n".join(state['case_summaries'] or [])
    messages = [
        SystemMessage(content=ARGUMENT_BUILDER_PROMPT),
        HumanMessage(content=f"Question: {state['legal_question']}\nPlan: {state['research_plan']}\n\n{content}")
    ]
    response = model.invoke(messages)
    return {"argument_draft": response.content, "revision_number": state.get("revision_number", 1) + 1}

def critique_node(state: AgentState):
    """Critique the legal argument."""
    messages = [
        SystemMessage(content=CRITIQUE_PROMPT),
        HumanMessage(content=state['argument_draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}

def should_continue(state):
    """Determine if more revisions are needed."""
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "critique_node"

# Build the state graph
builder = StateGraph(AgentState)
builder.add_node("plan", research_plan_node)
builder.add_node("find_cases", case_finder_node)
builder.add_node("summarize", case_summarizer_node)
builder.add_node("build_argument", argument_builder_node)
builder.add_node("critique_node", critique_node)
builder.set_entry_point("plan")

# Define the graph edges
builder.add_edge("plan", "find_cases")
builder.add_edge("find_cases", "summarize")
builder.add_edge("summarize", "build_argument")
builder.add_conditional_edges(
    "build_argument",
    should_continue,
    {END: END, "critique_node": "critique_node"}
)
builder.add_edge("critique_node", "build_argument")

# Compile the graph
graph = builder.compile(checkpointer=memory)

# Example usage
thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream({
    'legal_question': "What is the standard for summary judgment in federal court?",
    "max_revisions": 2,
    "revision_number": 1,
    "sources": [],
    "case_summaries": [],
}, thread):
    print(s)