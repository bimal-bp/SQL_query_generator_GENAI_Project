from langchain.llms import google_palm
from langchain.utilities import SQLDatabase 
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbedding 
from langchain.vectorstores import chroma 
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX,_mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from few_shots import few_shots 

import os 
from dotenv import load_dotenv 
load_dotenv()


def get_few_shot_db_chain():
    db_user="root"
    db_password="root"
    db_host="localhost"
    db_name="orbt_dresses"

    db=SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                            sample_rows_in_table_info=3)

    llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"],temperature=0.1)

    embeddings=HuggingFaceEmbedding(model_name='sentence-transformer/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstores=Chroma.from_texts(to_vectorize,embeddings,metadats=few_shots)

    example_selector= SemanticSimilarityExampleSelector(
        vectorstores=vectorstores,
        k=2
    )
    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".
    Use the following format:
    
    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    No pre-amble.
    """

    example_prompt  = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input","table_info","top_k"],
    )

    chain=SQLDatabaseChain.from_llm(llm,db,verbose=True,prompt=few_shot_prompt)
    return chain