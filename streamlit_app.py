import streamlit as st
import WithLangGraph, WithSampleSQLs
st.set_page_config(
    page_title="text to SQL",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "sql_helper" not in st.session_state:
    st.session_state.sql_helper = WithSampleSQLs.WithSampleSQLs()
    st.session_state.lang_graph = WithLangGraph.WithLangGraph()

st.title("Text to SQL")

with st.form("query_form", clear_on_submit=False):
    user_query = st.text_area("Message", key="user_query", max_chars=500)
    send_button = st.form_submit_button("Send")
    col1, col2 = st.columns([0.5, 0.5])
    if send_button and user_query:
        with col1:
            st.info("With RAG of sample SQLs")
            try:
                sql_query, metadata = st.session_state.sql_helper.get_sql(user_query)
                st.write(sql_query)
                st.write(metadata)
                st.write(st.session_state.sql_helper.run_query(sql_query))
            except BaseException as e:
                st.write(e)
        with col2:
            st.info("With Lang Graph")
            try:
                sql_query, metadata = st.session_state.lang_graph.get_sql(user_query)
                st.write(sql_query)
                st.write(metadata)
                if sql_query is not None:
                    st.write(st.session_state.lang_graph.run_query(sql_query))
                else:
                    st.write("Can only do select queries")
            except Exception as e:
                st.write(e)
