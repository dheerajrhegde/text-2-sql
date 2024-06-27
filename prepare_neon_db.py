import os, sqlalchemy, requests

repo_url = "https://raw.githubusercontent.com/neondatabase/mistral-neon-text-to-sql/main/data/"
fnames = ["northwind-schema.sql", "northwind-queries.jsonl"]

# Connect to the database
NEON_CONNECTION_STRING = os.getenv('NEON_CONNECTION_STRING')
engine = sqlalchemy.create_engine(
    url=NEON_CONNECTION_STRING, pool_pre_ping=True, pool_recycle=300
)

try:
    os.mkdir("data")
except FileExistsError:
    pass

for fname in fnames:
    response = requests.get(repo_url + fname)
    with open(f"data/{fname}", "w") as file:
        file.write(response.text)

# run the DDL script to create the database
with engine.connect() as conn:
    with open("data/northwind-schema.sql") as f:
        conn.execute(sqlalchemy.text(f.read()))
    conn.commit()


