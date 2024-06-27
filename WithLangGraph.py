from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import operator, os
from langchain.agents import tool
from langchain.pydantic_v1 import BaseModel, Field
import base64
from sqlalchemy_schemadisplay import create_schema_graph
from sqlalchemy import MetaData
from langchain_openai.chat_models import ChatOpenAI
import re, sqlalchemy
from tavily import TavilyClient
from langchain.adapters.openai import convert_openai_messages

NEON_CONNECTION_STRING = os.getenv('NEON_CONNECTION_STRING')
engine = sqlalchemy.create_engine(
    url=NEON_CONNECTION_STRING, pool_pre_ping=True, pool_recycle=300
)
model = ChatOpenAI(model='gpt-4o', openai_api_key=os.getenv("OPENAI_API_KEY"))

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        # print("Calling open AI")
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        # print("in exists action")
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            # print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

        # print("Back to the model!")
        return {'messages': results}

class SyntaxCheck(BaseModel):
    question: str = Field(..., description="Find a function or syntax in the PostgreSQL documentation")

@tool(args_schema=SyntaxCheck)
def find_sql_function(self, error_string: str) -> str:
    """
    Will help find the needed functions from Database Documentation to be used while building qquery
    """
    print("Calling find_function")
    tavily_tool = TavilySearchResults(max_results=4,
                                      search_depth="advanced",
                                      include_domains=["https://www.postgresql.org/docs/current/sql-syntax.html"]
                                      )
    return tavily_tool.invoke(error_string)

class ErrorMessage(BaseModel):
    error_string: str = Field(..., description="Error code or message to be checked against documentation")

@tool(args_schema=ErrorMessage)
def check_error(self, error_string: str) -> str:
    """
    Called with the error code and error description as the argument to get guidance on how to solve the error
    """
    print("Calling check_error")
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    content = client.search(error_string, search_depth="advanced")["results"]

    # setup prompt
    prompt = [{
        "role": "system",
        "content": f'You are an AI research assistant. ' \
                   f'Your sole purpose is to give guidance on how to address SQL errors'
    }, {
        "role": "user",
        "content": f'Information: """{content}"""\n\n' \
                   f'Using the above information, answer the following' \
                   f'query: "How to address the SQL error --> {error_string}"' \
        }]

    # run gpt-4
    lc_messages = convert_openai_messages(prompt)
    guidance = model.invoke(lc_messages).content
    return guidance

@tool
def get_schema():
    """
    Get the schema of the database to use for creating and running the SQL query.
    """
    """with open("database_schema_er_diagram.png", "rb") as f:
        base64_encoded = base64.b64encode(f.read()).decode("utf-8")
        base64_string_with_prefix = f"data:image/png;base64, {base64_encoded}"
        return base64_string_with_prefix"""
    NEON_CONNECTION_STRING = os.getenv('NEON_CONNECTION_STRING')
    engine = sqlalchemy.create_engine(
        url=NEON_CONNECTION_STRING, pool_pre_ping=True, pool_recycle=300
    )

    metadata = MetaData()
    metadata.reflect(bind=engine)

    graph = create_schema_graph(
        engine=engine,
        metadata=metadata,
        show_datatypes=True,  # Show column data types
        show_indexes=True,  # Show indexes (not relevant for all databases)
        rankdir='LR',  # Left to right layout
        concentrate=False  # Avoid node overlapping
    )

    return graph.create_plain()


class GeneratedSQL(BaseModel):
    sql_string: str = Field(..., description="SQL statement generated by the LLM")

@tool(args_schema=GeneratedSQL)
def update_delete_drop_insert(sql_string: str) -> str:
    """
    Will validate the SQL to ensure it is not an update, delete, drop or insert query
    """
    if re.search(r"(?i)(update|delete|drop|insert)", sql_string):
        return "The SQL query should not be an update, delete, drop or insert query."
    return "The SQL query is a select statement."

@tool(args_schema=GeneratedSQL)
def validate_sql(sql_string: str) -> str:
    """
    Will validate the SQL to ensure it is syntactically and functionally correct
    """
    print("Calling validate_sql")
    try:
        engine.connect().execute(sqlalchemy.text("EXPLAIN "+ sql_string))
        return "The SQL query is syntactically and functionally correct."
    except BaseException as e:
        return f"The SQL query is not correct. Error: {e}"

class WithLangGraph:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o")
        self.prompt = """
            You are an AI assistant that converts natural language questions into SQL queries. 
            To do this, you will be provided with three key pieces of information:
    
            The actual natural language question to convert into an SQL query:
            <question>
            {QUESTION}
            </question>
    
            Follow the instructions below:
            1. Your task is to generate an SQL query that will retrieve the data needed to answer the question, based on the database schema. 
            2. First, carefully study the provided schema  to understand the structure of the database and how the examples map natural language to SQL for this schema.
            3. You can only do select queries. Incase of any other queries, you should return a SQL statement that prints "only select query allowed" in the <sql> tag in step 4
            4. You should validate the SQL you have created for syntax and function errors. If there are any errors, you should go back and address the error in the SQL. Recreate the SQL based by addressing the error. 
            5. Your answer should have two parts: 
            - Inside <scratchpad> XML tag, write out step-by-step reasoning to explain how you are generating the query based on the schema, example, and question. 
            - Then, inside <sql> XML tag, output your generated SQL. 
        """
        NEON_CONNECTION_STRING = os.getenv('NEON_CONNECTION_STRING')
        self.engine = sqlalchemy.create_engine(
            url=NEON_CONNECTION_STRING, pool_pre_ping=True, pool_recycle=300
        )
        self.tools = [update_delete_drop_insert, check_error, get_schema, validate_sql]

    def get_sql(self, question):
        prompt = self.prompt.format(QUESTION=question)


        abot = Agent(self.model, self.tools, system=prompt)
        thread = {"configurable": {"thread_id": "1"}}

        response = abot.graph.invoke(
            {"messages": [HumanMessage(content=[{"type": "text", "text": question}])], "thread": thread})
        print(response)
        if response.get("messages")[-1].content == "Can only do select queries":
            return None
        sql_query = re.search(r"<sql>(.*?)</sql>", response.get("messages")[-1].content, re.DOTALL).group(1)
        return sql_query, response.get("messages")[-1].response_metadata["token_usage"]

    def run_query(self, sql_query):
        with self.engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(sql_query))
            return result.fetchall()
            """for row in result:
                print(row._mapping)"""




if __name__ == "__main__":
    sql_helper = WithLangGraph()
    """question = "Find the employee who has processed the most orders and display their \
    full name and the number of orders they have processed?" """
    """question = "create a new customer with company name as 'ABC Corp' and contact name as 'John Doe'" """
    question = "Give me the region description (eastern, western) where the order has icnreasaed the most between 1997 and 1998. Give the counts and percentage increase."
    sql_query, metadata = sql_helper.get_sql(question)
    print(sql_query)
    print(metadata)
    if sql_query is not None:
        print(sql_helper.run_query(sql_query))


