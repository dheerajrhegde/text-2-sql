import os, sqlalchemy
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
import re
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_core.messages import HumanMessage
import time, json
class WithSampleSQLs:
    def __init__(self):
        self.NEON_CONNECTION_STRING = os.getenv('NEON_CONNECTION_STRING')
        self.engine = sqlalchemy.create_engine(
            url=self.NEON_CONNECTION_STRING, pool_pre_ping=True, pool_recycle=300
        )
        embeds_model = OpenAIEmbeddings()
        self.vector_store = PGVector(
            embeddings=embeds_model,
            connection=self.engine,
            use_jsonb=True,
            collection_name="text-to-sql-context",
        )


        # DDL statements to create the Northwind database

        _all_stmts = []
        with open("./data/northwind-schema.sql", "r") as f:
            stmt = ""
            for line in f:
                if line.strip() == "" or line.startswith("--"):
                    continue
                else:
                    stmt += line
                    if ";" in stmt:
                        _all_stmts.append(stmt.strip())
                        stmt = ""

        ddl_stmts = [x for x in _all_stmts if x.startswith(("CREATE", "ALTER"))]

        docs = [
            Document(page_content=stmt, metadata={"id": f"ddl-{i}", "topic": "ddl"})
            for i, stmt in enumerate(ddl_stmts)
        ]

        self.vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])

        # Sample question-query pairs

        with open("./data/northwind-queries.jsonl", "r") as f:
            docs = [
                Document(
                    page_content=pair,
                    metadata={"id": f"query-{i}", "topic": "query"},
                )
                for i, pair in enumerate(f)
            ]

        start = 0
        end = 5

        for doc in docs:
            print(doc.metadata["id"])
            self.vector_store.add_documents([doc], ids=[doc.metadata["id"]])
            time.sleep(0.5)

        #self.vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])

    def get_sql(self, question):
        similar_queries = self.vector_store.similarity_search(
            query=question, k=3, filter={"topic": {"$eq": "query"}}
        )

        relevant_ddl_stmts = self.vector_store.similarity_search(
            query=question, k=5, filter={"topic": {"$eq": "ddl"}}
        )

        schema = ""
        for stmt in relevant_ddl_stmts:
            schema += stmt.page_content + "\n\n"

        examples = ""
        for stmt in similar_queries:
            text_sql_pair = json.loads(stmt.page_content)
            examples += "Question: " + text_sql_pair["question"] + "\n"
            examples += "SQL: " + text_sql_pair["query"] + "\n\n"

        prompt = """
        You are an AI assistant that converts natural language questions into SQL queries. To do this, you will be provided with three key pieces of information:

        1. Some DDL statements describing tables, columns and indexes in the database:
        <schema>
        {SCHEMA}
        </schema>

        2. Some example pairs demonstrating how to convert natural language text into a corresponding SQL query for this schema:  
        <examples>
        {EXAMPLES}
        </examples>

        3. The actual natural language question to convert into an SQL query:
        <question>
        {QUESTION}
        </question>

        Follow the instructions below:
        1. Your task is to generate an SQL query that will retrieve the data needed to answer the question, based on the database schema. 
        2. First, carefully study the provided schema and examples to understand the structure of the database and how the examples map natural language to SQL for this schema.
        3. Your answer should have two parts: 
        - Inside <scratchpad> XML tag, write out step-by-step reasoning to explain how you are generating the query based on the schema, example, and question. 
        - Then, inside <sql> XML tag, output your generated SQL. 
        """

        chat_model = ChatOpenAI()
        prompt = prompt.format(QUESTION=question, SCHEMA=schema, EXAMPLES=examples)

        response = chat_model.invoke(prompt)
        sql_query = re.search(r"<sql>(.*?)</sql>", response.content, re.DOTALL).group(1)
        return sql_query, response.response_metadata["token_usage"]

    def run_query(self, sql_query):
        with self.engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(sql_query))
            return result.fetchall()


if __name__ == "__main__":
    sql_helper = WithSampleSQLs()

    question = "Find the employee who has processed the most orders and display their \
        full name and the number of orders they have processed?"
    """question = "create a new customer with company name as 'ABC Corp' and contact name as 'John Doe'" """

    sql_query, metadata = sql_helper.get_sql(question)
    print(sql_query)
    print(metadata)
    print(sql_helper.run_query(sql_query))
