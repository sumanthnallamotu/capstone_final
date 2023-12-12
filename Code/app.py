import streamlit as st
import boto3
import json
import ast

system_prompt = "You are an excellent assistant AI. Please answer any questions."

lambda_client = boto3.client(
    'lambda',
    aws_access_key_id = "ASIA354PVNRIURZUISXA",
    aws_secret_access_key = "kMtMCdXfIDoxm1lLPmby9YbEtIDNLvs4cxckGadP",
    aws_session_token = "FwoGZXIvYXdzEEcaDA0lHwZLb4eH4rJEfSJu8uy+QdjnZaaWQfjKXTXrbEK1jPkmXCDK49g4GcaAiw74Adz1oTAIVNB4oeNrLHIMu8otxkfUVvyusiJUtQTdqh0YekNMyLDz6qaJzISl+ugC7mgZaCEo4PQMzm4fkW7qmZHrmxOPfI489eX78kwoz67FqwYyKGUU/ppii59mMI3A/1mSbB6j9H3zB9MnaweskujFnb/8LeaBL++Tnk8="
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": system_prompt}
        ]


def communicate():
    messages = st.session_state["messages"]

    user_message = {"role": "user", "content": st.session_state["user_input"]}
    messages.append(user_message)

    lambda_input = {"query": st.session_state["user_input"]}

    response = lambda_client.invoke(
        FunctionName = 'capstone_func',
        Payload = json.dumps(lambda_input)
    )

    lambda_response = response['Payload'].read().decode("utf-8")
    resp = ast.literal_eval(lambda_response)
    print(resp)
    resp = resp['body']
    new_resp = resp.replace('\\n', '  \n')
    print(new_resp)

    bot_message = {"role": "assistant", "content": new_resp}
    messages.append(bot_message)

    st.session_state["user_input"] = ""



st.title("Interactive Predictive Maintenance- Powered by LLM")
st.subheader("by Sumanth Nallamotu")
st.sidebar.markdown("## GWU Fall 2023 DATS Capstone")
st.sidebar.markdown("Welcome to my capstone demo! Today, I will be demonstrating my \
                    Interactive Predictive Maintenance application.  \n LLM's used:  \n- defog/sqlcoder-34B-alpha  \n- meta/llama2-13b-chat\
                    \n   \n This is a completely open-source implementation hosted on AWS, with no third-party dependencies like OpenAI or LangChain. \
                    Basically, the front-end interacts with an AWS Lambda function that passes the input through \
                    two SageMaker endpoints hosting the above models. The queries are run on Athena tables containing \
                    historic simulated industrial machine data and forecasted risk metrics. \
                    As you'll see in the results section, sqlcoder-34B-alpha outperformed GPT-4 with regard to text-to-SQL capabilities in my experiments. \
                    Feel free to walk through the rest of the poster and make sure to \
                    scan the QR code or directly email me at sumanthn@gwmail.gwu.edu to send me any questions you might have. I'll be \
                    emailing responses to everyone! Or, if you just want to chat about LLM's, please do not hesitate!")

user_input = st.text_input("Ask me questions about your data.", key="user_input", on_change=communicate)

if st.session_state["messages"]:
    messages = st.session_state["messages"]

    for message in reversed(messages[1:]):
        speaker = "User"
        if message["role"]=="assistant":
            speaker="Assistant"


        if speaker == "User":
            st.markdown(f'<h1 style="background-color:#C6DCF6;color:#000000;font-size:24px;">{speaker}: {str(message["content"])}</h1>', unsafe_allow_html=True)
        else:
            st.markdown(f'<h1 style="background-color:#FFC9AA;color:#000000;font-size:24px;">{speaker}: {str(message["content"])}</h1>', unsafe_allow_html=True)