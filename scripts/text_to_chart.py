from dotenv import load_dotenv
load_dotenv()


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def __df_metadata(df):
    """Generate schema for pandas dataframe."""
    schema = {}
    for i in zip(df.columns, df.dtypes):
        schema[i[0]] = str(i[1])
    return str(schema)


def generate_seaborn_code(df, user_prompt):
    """Generate seaborn code using LLMs."""
    schema = __df_metadata(df)
    
    # Define LLM and prompt
    llm = ChatOpenAI(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class python programmer and expert in using Seaborn libraries."),
        ("user", """Assume you have a data frame `df` with the following schema: {schema}. {input}. Dont create your own data, only give code as the output""")
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    result = chain.invoke({"schema": schema, "input": user_prompt})
    parsed_code = result.replace("```python", "").replace("```", "")

    return parsed_code


def run_generated_code(df, parsed_code):
    """Execure generated code."""
    # TODO: Validate code and add guardrails
    exec(parsed_code, {"df":df})


def generate_visualization(df, user_prompt):
    """Generate chart using user's data frame and prompt."""
    parsed_code = generate_seaborn_code(df, user_prompt)
    run_generated_code(df, parsed_code)

