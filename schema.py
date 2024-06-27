from sqlalchemy_schemadisplay import create_schema_graph
from sqlalchemy import MetaData
import sqlalchemy, os

NEON_CONNECTION_STRING = os.getenv('NEON_CONNECTION_STRING')
print(NEON_CONNECTION_STRING)

NEON_CONNECTION_STRING = "postgresql://neondb_owner:BKx0kHXUJY4V@ep-wild-lake-a5scrazo.us-east-2.aws.neon.tech/neondb"
engine = sqlalchemy.create_engine(
    url=NEON_CONNECTION_STRING, pool_pre_ping=True, pool_recycle=300
)

metadata = MetaData()
metadata.reflect(bind=engine)

graph = create_schema_graph(
    engine=engine,
    metadata=metadata,
    show_datatypes=True,  # Show column data types
    show_indexes=True,    # Show indexes (not relevant for all databases)
    rankdir='LR',         # Left to right layout
    concentrate=False     # Avoid node overlapping
)

# Output file
output_file = 'database_schema_er_diagram.png'

# Save the schema graph to a file
graph.write_png(output_file)
print(dir(graph))
print(graph.create_plain())