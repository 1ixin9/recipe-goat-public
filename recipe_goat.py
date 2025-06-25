from langchain.prompts import PromptTemplate
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import threading
from langchain_community.llms import Tongyi
import os
import time

# use ur own keys
from keys import qianfan_ak, qianfan_sk, def_keys

def_keys()

# models
llm = QianfanChatEndpoint(model="ERNIE-4.0-8K", streaming=True,
                          qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk, penalty_score=1)

llm2 = Tongyi(model_name="qwen-max")

embed = QianfanEmbeddingsEndpoint(
    model="bge_large_zh", endpoint="bge_large_zh", qianfan_ak=qianfan_ak, qianfan_sk=qianfan_sk)

# list of all the prompts we use
prompt1 = PromptTemplate(
    template="""
    You are a cooking assistant for recipe writer and you specialize in ingredient research.
    Consider the ingredient {prod}'s key features and consumer reviews.
    Detail the ingredient's unique points and qualities.
    Please provide insights on how the ingredient appeals to the general public. \n\n
    Use the following information to come up with your answer: \n
    {context}""",
    input_variables=["prod", "context"]
)

prompt2 = PromptTemplate(
    template="""
    You are a cooking assistant for recipe writer and you specialize in ingredient flavor research.
    Consider the ingredient {prod}'s key flavor features and flavor reviews.
    Please provide insights on how the ingredient's flavor appeals to the general public. \n\n
    Use the following information to come up with your answer: \n
    {context}""",
    input_variables=["prod", "context"]
)

prompt3 = PromptTemplate(
    template="""
    You are a cooking assistant for recipe writer and you specialize in ingredient flavor compatibility research.
    Consider the ingredient {prod}'s key flavors and how it would best fit a {flavor} dish.
    Please provide insights on how the ingredient may either highlight, complement, or star in a {flavor} dish. \n\n
    Use the following information to come up with your answer: \n
    {context}""",
    input_variables=["prod", "flavor", "context"]
)

prompt4 = PromptTemplate(
    template="""
    You are a cooking assistant for recipe writer and you specialize in meal creation research.
    Consider the ingredient {prod}'s key qualities and how it might fit as a {meal} meal.
    Please provide insights on how the ingredient may either highlight, complement, or star as a {meal} dish. \n\n
    Use the following information to come up with your answer: \n
    {context}""",
    input_variables=["prod", "meal", "context"]
)


def split_text(text):
    words = text.split()
    chunks, current_chunk = [], []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= 400 and word:
            current_chunk.append(word)
        elif current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def rag_search(query):
    url = "https://www.google.com/search?q="

    search_query = {'wd': query}

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, params=search_query, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')

    results, content = [], []

    for item in soup.find_all('div', class_='result'):
        link = item.find('a', href=True)  # 'a' is a link notation
        if link:
            results.append(link['href'])

    docs = get_page(results)

    for doc in docs:
        page_text = re.sub("\n\n+", "\n", doc)

        if page_text:
            content.append(page_text)

    return content


def get_page(urls):
    docs = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    for url in urls:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            page_text = "\n".join([p.get_text() for p in paragraphs])
            chunks = split_text(page_text)
            docs.extend(chunks)

    return docs


def process_search(query):

    q_embed = embed.embed_query(query)
    search_result = rag_search(query)

    if not search_result:
        return "no relevant content found"

    all_chunks = []
    for result in search_result:
        chunks = split_text(result)
        all_chunks.extend(chunks)

    if not all_chunks:
        return "no valid chunks in search"

    search_embed = []
    i1, i2 = 0, 0

    while i2 < len(all_chunks) - 1:
        i2 += 1
        search_embed.extend(embed.embed_documents(all_chunks[i1:i2]))
        i1 += 1

    if not search_embed:
        return "couldn't embed search results"

    search_embed = np.array(search_embed)
    similarity_scores = np.dot(search_embed.T, q_embed) / (
        np.linalg.norm(search_embed.T, axis=1) * np.linalg.norm(q_embed))
    filtered_results = [(result, score) for result, score in zip(
        search_result, similarity_scores) if score > 0.75]

    max_ctxt = 3
    if len(filtered_results) < 3:
        max_ctxt = len(filtered_results)
    top_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[
        :max_ctxt]

    rag_results = " ".join([result[0] for result in top_results])

    return rag_results


def re_search(query):
    promptV2 = PromptTemplate(
        template="""    
        You are a search engine input writer that optimizes queries for online searching.\n     
        Optimize the following query to get the best search results and only output the result query:\n    
        {query}     
        """,
        input_variables=["query"]
    )

    rewriter = promptV2 | llm | StrOutputParser()

    new_query = rewriter.invoke({"query": query})

    return process_search(new_query)


def crag_search(llm, prompt, ingredient, keywords, meal=None, flavor=None):
    query = ingredient + keywords
    if meal:
        query = query + " " + meal
    if flavor:
        query = query + " " + flavor

    search_result = process_search(query)

    grade_prompt = PromptTemplate(
        template="""    
        Determine whether the following search results are relevant to the query\n   
        Search results: \n\n {result} 
        \n\n Query: {query}
        \n\nRespond with only either 'yes' or 'no':
        """,
        input_variables=["query", "result"]
    )

    retrieval_grader = grade_prompt | llm | StrOutputParser()

    result = retrieval_grader.invoke(
        {"query": query, "result": search_result})

    if result == "no":
        search_result = re_search(query)

    expertGPT = prompt | llm | StrOutputParser()

    return expertGPT.invoke(
        {"prod": ingredient, "flavor": flavor, "meal": meal, "context": search_result})


def call_reciGPT(ingredient, meal, flavor):

    prod_des = "伊利羊奶粉"
    season = "万圣节"

    context = []

    # defines threading function
    def call_crag(llm, prompt, ingredient, keywords, meal=None, flavor=None):
        ctxt = crag_search(llm, prompt, ingredient, keywords, meal, flavor)
        context.append(ctxt)

    # initializes threads
    t1 = threading.Thread(target=call_crag, args=(
        llm, prompt1, ingredient, meal, flavor, " review quality what is "+ingredient))
    t2 = threading.Thread(target=call_crag, args=(
        llm2, prompt2, ingredient, meal, flavor, " taste flavor texture review "))
    t3 = threading.Thread(target=call_crag, args=(
        llm, prompt3, ingredient, meal, flavor, " with "))
    t4 = threading.Thread(target=call_crag, args=(
        llm2, prompt4, ingredient, meal, flavor, " for "))

    t1.start()
    time.sleep(2)
    t2.start()
    time.sleep(2)
    t3.start()
    time.sleep(2)
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()

    prompt5 = PromptTemplate(
        template="""You are an educated and creative recipe writer. You take in a specific ingredient 
        and a specific type of meal (breakfast/lunch/etc.) and a key flavor (salty/sweet/etc.) 
        in order to write a recipe that matches the meal type, key flavor, and features the ingredient 
        as its key ingredient.\n\n

        Key Ingredient: {prod}\n\n
        Meal Type: {meal}\n\n
        Key Flavor: {flavor}\n\n
        
        Primarily use the following information to come up with your recipe:

        Key Ingredient qualities: {context1}\n
        Flavor and texture profile of key ingredient: {context2}\n
        Key Ingredient and key flavor compatibility: {context3}\n
        Key ingredient and meal type compatibility: {context4}\n\n
        
        Make sure the recipe is exactly what the desired meal type is.
        If the meal is a dinner but the key ingredient doesn't fit a traditional dinner,
        use the key ingredient creatively in a dinner recipe.
        Make sure your recipe is formatted exactly like the following:\n\n
        
        Recipe Name:\n
        Description:\n\n
        
        Ingredients:\n\n
        
        Preparation:\n\n
        
        Instructions:\n
        
        Do not separate the ingredients, preparation, or instructions section even if there are multiple components to the recipe.
        Make sure everything is under their specific section.
        The ingredients should only be separated by one line each at MOST.
        Make sure all of the titles to each section are correctly placed.
        An example would be:\n\n
        
        Recipe Name: Banana Pie\n
        Description: A delicious banana pie.\n\n
        
        Ingredients:\n
        1. 2 bananas\n
        2. etc.\n\n
        
        Preparation:\n
        1. Step 1\n
        2. etc.\n\n
        
        Instructions:\n
        1. Step 1\n
        2. etc.
        """,
        input_variables=["prod", "context1", "context2",
                         "context3", "context4", "meal", "flavor"]
    )

    reciGPT = prompt5 | llm | StrOutputParser()

    try:
        ans = reciGPT.invoke(
            {"prod": ingredient, "context1": context[0], "context2": context[1],
             "context3": context[2], "context4": context[3], "meal": meal, "flavor": flavor})
    except Exception as e:
        print("error: ", e)
        return

    return ans


def parse_recipe(ingredient, meal, flavor):

    output = call_reciGPT(ingredient, meal, flavor)

    patterns = {
        "Recipe Name": r"Recipe Name:\s*(.*?)\n",
        "Description": r"Description:\s*(.*?)\n\n",
        "Ingredients": r"Ingredients:\n\s*(.*?)\n\n",
        "Preparation": r"Preparation:\n\s*(.*?)\n\n",
        "Instructions": r"Instructions:\n\s*(.*)"
    }

    recipe_dict = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.DOTALL)
        if match:
            recipe_dict[key] = match.group(1).strip()

    return recipe_dict
