from flask import Flask, render_template, request, jsonify, redirect, url_for

import os
import openai
openai.api_key = os.environ["openai_api"]

import requests
from googletrans import Translator
from bs4 import BeautifulSoup
import urllib.parse
import html
import re
from urllib.parse import urlparse
from uuid import uuid4
from scipy.spatial import distance
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from time import sleep
import validators

tokenizer = tiktoken.get_encoding('p50k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

translator = Translator()

def scrape(url, domain):

    if '.pdf' in url:
        return None
    else:
        res = requests.get(url)
        if res.status_code != 200:
            print(f"{res.status_code} for '{url}'")
            return None
        soup = BeautifulSoup(res.text, 'html.parser')

        # Find all links to local pages on the website
        local_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith(domain) or href.startswith('./') \
                or href.startswith('/') or href.startswith('modules') \
                or href.startswith('use_cases'):
                local_links.append(urllib.parse.urljoin(domain, href))

        # # Find the main content using CSS selectors
        # main_content = soup.select('body main')[0]

        # # Extract the HTML code of the main content
        # main_content_html = str(main_content)

        # Extract the plaintext of the main content
        main_content_text = soup.get_text() # main_content.get_text()

        # Remove all HTML tags
        main_content_text = re.sub(r'<[^>]+>', '', main_content_text)

        # Remove extra white space
        main_content_text = ' '.join(main_content_text.split())

        # Replace HTML entities with their corresponding characters
        main_content_text = html.unescape(main_content_text)

        # Translate all text to english
        new_text = translator.translate(main_content_text[:4500]).text

        # return as json
        return {
            "url": url,
            "text": new_text
        }, local_links
    

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form["name"]
        validation = validators.url(url)

        if validation:
            links = [url]
            scraped = set()
            data = []

            domain = urlparse(links[0]).netloc
            domain = 'https://'+ domain + '/'

            while True:
                if len(links) == 0:
                    print("Complete")
                    break
                url = links[0]
                res = scrape(url, domain)
                scraped.add(url)
                # Set num of pages to be scraped
                if len(scraped) == 2:
                    break
                if res is not None:
                    page_content, local_links = res
                    data.append(page_content)
                    # add new links to links list
                    links.extend(local_links)
                    # remove duplicates
                    links = list(set(links))
                # remove links already scraped
                links = [link for link in links if link not in scraped]


            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=20,
                length_function=tiktoken_len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = []

            for idx, record in enumerate(data):
                texts = text_splitter.split_text(record['text'])
                chunks.extend([{
                    'id': str(uuid4()),
                    'text': texts[i],
                    'chunk': i,
                    'url': record['url']
                } for i in range(len(texts))])

            embed_model = "text-embedding-ada-002"

            batch_size = 100  # how many embeddings we create and insert at once

            for i in range(0, len(chunks), batch_size):
                # find end of batch
                i_end = min(len(chunks), i+batch_size)
                meta_batch = chunks[i:i_end]
                # get ids
                ids_batch = [x['id'] for x in meta_batch]
                # get texts to encode
                texts = [x['text'] for x in meta_batch]
                # create embeddings (try-except added to avoid RateLimitError)
                try:
                    res = openai.Embedding.create(input=texts, engine=embed_model)
                except:
                    done = False
                    while not done:
                        sleep(5)
                        try:
                            res = openai.Embedding.create(input=texts, engine=embed_model)
                            done = True
                        except:
                            pass
                embeds = [record['embedding'] for record in res['data']]
                # cleanup metadata
                meta_batch = [{
                    'text': x['text'],
                    'chunk': x['chunk'],
                    'url': x['url']
                } for x in meta_batch]
                to_upsert = list(zip(ids_batch, embeds, meta_batch))
            
            print(len(to_upsert))

            global vectors
            global all_texts

            vectors = []
            for a,b,c in to_upsert:
                vectors.append(b)

            all_texts = []
            for a,b,c in to_upsert:
                all_texts.append(c['text'])
            
            print(len(vectors), len(all_texts))

            return redirect('/goo')
        else:
            return render_template('chat4.html', response='Enter a valid URL')
    
    return render_template('chat4.html', response='')

@app.route("/goo")
def goo():
    return render_template('chat.html')


@app.route("/submit", methods=["GET", "POST"])
def submit():
    msg = request.json['message']
    return jsonify({'answer': "hello"+msg})


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    return get_Chat_response(msg)


def get_Chat_response(text):

    embed_model = "text-embedding-ada-002"
    res = openai.Embedding.create(
        input=[text],
        engine=embed_model
    )

    # Embed the query
    xq = res['data'][0]['embedding']

    cosines = []
    for i,vec in enumerate(vectors):
        cosines.append(1 - distance.cosine(xq, vec))
    
    results = sorted(range(len(cosines)), key=lambda i: cosines[i], reverse=True)[:3]

    contexts = []
    for i in results:
        contexts.append(all_texts[i])

    augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+text
    
    # system message to 'prime' the model
    primer = f"""You are Q&A customer support bot for a company. A highly intelligent system that answers
    user questions based on the information provided by the user above
    each question. If the information can not be found in the information
    provided by the user you truthfully say "Sorry I cannot find answer to your question".
    """

    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_query}
        ]
    )

    answer = res['choices'][0]['message']['content']

    return answer


if __name__ == '__main__':
    app.run(debug=True)
