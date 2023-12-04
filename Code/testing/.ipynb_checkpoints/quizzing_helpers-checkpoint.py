import pandas as pd
import numpy as np
import datetime

import os
import openai

import awswrangler as wr

import time
import boto3

import requests
import json

sql_endpoint = "https://yhkqe6os38yw017b.us-east-1.aws.endpoints.huggingface.cloud"

headers = {
	"Authorization": "Bearer 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",
	"Content-Type": "application/json"
}

# Helper functions

def get_ddl(athena_client, table_name):
    queryStart = athena_client.start_query_execution(
        QueryString = "SHOW CREATE TABLE {}".format(table_name),
        QueryExecutionContext = {
            'Database': 'capstone_v3'
        },
        ResultConfiguration = { 'OutputLocation': 's3://aws-athena-query-results-820103179345-us-east-1/text2sql/'}
    )
    time.sleep(5)
    queryExecution = athena_client.get_query_execution(QueryExecutionId=queryStart['QueryExecutionId'])
    results = athena_client.get_query_results(QueryExecutionId=queryStart['QueryExecutionId'])
        
    ddl = ""
    for i in range(len(results['ResultSet']['Rows'])):
        ddl += results['ResultSet']['Rows'][i]['Data'][0]['VarCharValue']
    ddl = ddl.split(')')[0]
    
    return ddl


def ask_gpt(openai_client, athena_client, question):
    """ ask GPT-4 some questions
    """
    
    start = time.time()
    response = openai_client.chat.completions.create(
        model='gpt-4',
        messages=[
            {
                "role": "system",
                "content": "Given the following SQL tables, your job is to write queries given a user's request. If the question doesn't match the schemas, tell the user. If the query matches the the schemas, only return SQL queries- no extra text. datetimes must be included. If the query involves dates, make sure you cast the date properly so the comparison works. Example: date 'YYYY-MM-DD' \n {} {};".format(get_ddl(athena_client,'predictions_prod'), get_ddl(athena_client,'telemetry_extended_v3'))
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        temperature=0.1,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

    output_query = response.choices[0].message.content
    gpt_time = time.time() - start
    
    context = "Here is the context: We are looking at industrial factory machine data. You will be provided a string representation of a dataframe with SQL query results. Your job is to summarize thes results and give recommendations for how to address these issues. Here is the corresponding SQL query: " + output_query
    
    athena_result = ""
    
    start = time.time()
    try:
        df = wr.athena.read_sql_query(
            sql=output_query,
            database='capstone_v3',
            ctas_approach=True)
        athena_result = str(df.head(15))
    except:
        athena_result = "The query could not be processed by Athena!"
        
    query_success = 0
    
    if athena_result != "The query could not be processed by Athena!":
        query_success = 1
        
    query_time = time.time() - start
        
    
    start = time.time()
    response = openai_client.chat.completions.create(
        model='gpt-4',
        messages=[
            {
                "role": "system",
                "content": context
            },
            {
                "role": "user",
                "content": "Evaluate the following output: {}. If the query was successful, interpret the dataframe (i.e. top records). If the query failed, communicate that to the user:".format(athena_result)
            }
        ],
        temperature=0.1,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    
    inter_time = time.time() - start

    return {
        "LLM": "GPT-4",
        "Question": question,
        "Query": output_query,
        "Interpretation": response.choices[0].message.content,
        "Text-to-SQL Time": np.round(gpt_time,2),
        "Query Success": query_success,
        "Query Time": np.round(query_time,2),
        "Interpretation Time": np.round(inter_time,2)
    }

def query(sql_endpoint, headers, payload):
	response = requests.post(sql_endpoint, headers=headers, json=payload)
	return response.json()

def ask_7b(llama_endpoint, athena_client, sagemaker_client, question):
    """ ask sqlcoder_7b some questions
    """
    
    prompt = """### Task
    Generate a SQL query to answer the following question:
    `{question}`

    ### Database Schema
    This query will run on databases whose schemas are represented in this string:
    `{pred_ddl}`;

    `{telemetry_ddl}`;

    -- the prediction table's columns each represent the predictions of a vehicle's risk scores
    -- the telemetry table contains a timeseries of machines along with metrics collected for each record
    -- If the query involves dates, make sure you cast the date properly so the comparison works. Example: date 'YYYY-MM-DD'
    ;

    ### SQL
    Given the database schema, here is the SQL query that answers `{question}`:
    ```sql
    """.format(question=question, pred_ddl=get_ddl(athena_client,'predictions_prod'), telemetry_ddl=get_ddl(athena_client,'telemetry_extended_v3'))
    
    start = time.time()
    output = query(sql_endpoint, headers,{
        "inputs": prompt,
        "parameters": {'max_new_tokens': 500, "top_p": 0.1, "temperature": 0.2}
    })
    
    output_query = output[0]['generated_text']
    output_query = output_query.replace('\n','')
    output_query = output_query.replace('```','')
    output_query = output_query.replace("' '","")
    
    text2sql_time = time.time() - start
    
    start = time.time()
    
    query_success = 0
    
    try:
        df = wr.athena.read_sql_query(
            sql=output_query,
            database='capstone_v3',
            ctas_approach=True)
        athena_result = str(df)
        query_success = 1
    except:
        athena_result = "The query could not be processed by Athena!"
        
        
        
    query_time = time.time() - start
    

    
    input_prompt = f"""[INST] <<SYS>>
    Your job is to interpret the results from the following query results: {athena_result}. If the query failed, just say that and nothing else. If the query didn't fail, provide recommendations on how to to fix the associated issues with the metrics returned.
    <</SYS>>{question} [/INST]"""
    

    payload = {
        "inputs": input_prompt,
        "parameters": {"max_new_tokens": 500, "top_p": 0.1, "temperature": 0.5}
    }
    
    start = time.time()
    response = sagemaker_client.invoke_endpoint(
        EndpointName=llama_endpoint,
        Body=json.dumps(payload),
        ContentType='application/json')
    inter_time = time.time() - start
    
    result = json.loads(response['Body'].read().decode())
    result = result[0]['generated_text'].split('[/INST]')[1]
    
    return {
        "LLM": "sqlcoder-7b",
        "Question": question,
        "Query": output_query,
        "Interpretation": result,
        "Text-to-SQL Time": np.round(text2sql_time,2),
        "Query Success": query_success,
        "Query Time": np.round(query_time,2),
        "Interpretation Time": np.round(inter_time,2)
    }

def ask_34b(llama_endpoint, athena_client, sagemaker_client, question):
    """ ask the sqlcoder_34b endpoint some questions
    """
    
    prompt = """### Task
    Generate a SQL query to answer the following question:
    `{question}`

    ### Database Schema
    This query will run on a database whose schema is represented in this string:
    `{pred_ddl}`;

    `{telemetry_ddl}`;

    -- the prediction table's columns each represent the predictions of a vehicle's risk scores
    -- the telemetry table contains a timeseries of machines along with metrics collected for each record
    -- If the query involves dates, make sure you cast the date properly so the comparison works. Example: date 'YYYY-MM-DD'
    ;

    ### SQL
    Given the database schema, here is the SQL query that answers `{question}`:
    ```sql
    """.format(question=question, pred_ddl=get_ddl(athena_client,'predictions_prod'), telemetry_ddl=get_ddl(athena_client,'telemetry_extended_v3'))
    
    start = time.time()
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 500, "top_p": 0.1, "temperature": 0.5}
    }
    
    response = sagemaker_client.invoke_endpoint(
        EndpointName="text2sql-34b",
        Body=json.dumps(payload),
        ContentType='application/json')
    
    result = json.loads(response['Body'].read().decode())
    output_query = result[0]['generated_text'].split('```sql')[1].split('```')[0]
    
    text2sql_time = time.time() - start
    
    start = time.time()
    
    query_success = 0
    
    try:
        df = wr.athena.read_sql_query(
            sql=output_query,
            database='capstone_v3',
            ctas_approach=True)
        athena_result = str(df.head(15))
        query_success = 1
    except:
        athena_result = "The query could not be processed by Athena!"
        
        
        
    query_time = time.time() - start
    

    
    input_prompt = f"""[INST] <<SYS>>
    Your job is to interpret the results from the following query results: {athena_result}. If the query failed, just say that and nothing else. If the query didn't fail, provide recommendations on how to to fix the associated issues with the metrics returned.
    <</SYS>>{question} [/INST]"""
    

    payload = {
        "inputs": input_prompt,
        "parameters": {"max_new_tokens": 500, "top_p": 0.1, "temperature": 0.5}
    }
    
    start = time.time()
    response = sagemaker_client.invoke_endpoint(
        EndpointName=llama_endpoint,
        Body=json.dumps(payload),
        ContentType='application/json')
    inter_time = time.time() - start
    
    result = json.loads(response['Body'].read().decode())
    result = result[0]['generated_text'].split('[/INST]')[1]
    
    return {
        "LLM": "sqlcoder-34b",
        "Question": question,
        "Query": output_query,
        "Interpretation": result,
        "Text-to-SQL Time": np.round(text2sql_time,2),
        "Query Success": query_success,
        "Query Time": np.round(query_time,2),
        "Interpretation Time": np.round(inter_time,2)
    }