import os
import time
import re
import chardet
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from threading import Timer
from uuid import uuid4
from tqdm.auto import tqdm
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import pinecone
from langchain.document_loaders import UnstructuredURLLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask_login import LoginManager, login_required, login_user, logout_user, UserMixin
import tempfile
from urllib.parse import urljoin
from urllib.parse import urlparse
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from flask_caching import Cache
from flask import current_app
from fpdf import FPDF
from urllib.parse import quote as url_quote
from html import escape
from docx import Document
import PyPDF2
from langchain.document_loaders import PyPDFium2Loader
import json
from flask import Flask, session, jsonify, request
import logging
from datetime import datetime
from html import escape
import uuid
from threading import Thread
import redis
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required, current_user
from flask_migrate import Migrate, upgrade
from werkzeug.security import generate_password_hash, check_password_hash
import logging
import uuid
from flask import Flask, request, redirect, url_for, flash, render_template
from flask_session import Session


# Laad configuratie uit een JSON-bestand
with open('config.json', 'r') as file:
    config = json.load(file)

# API-sleutels en instellingen
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Initialiseer de Redis-client
redis_url = "redis://red-cleho5c15k1s73acpmeg:6379"
redis_client = redis.StrictRedis.from_url(redis_url, decode_responses=True)


CHARACTER_LIMIT = config['character_limit']
MONTHLY_LIMIT = config['monthly_limit']

# Globale variabelen
DEFAULT_PROMPT_TEMPLATE = config['prompt_template']
CHARACTER_LIMIT = config['character_limit']
MONTHLY_LIMIT = config['monthly_limit']

# Definieer de standaardwaarden
DEFAULT_CHARACTER_COUNTER = 0
DEFAULT_PROMPT_TEMPLATE = config['prompt_template']
DEFAULT_WELCOME_MESSAGE = "ffff"
DEFAULT_TITLE_MESSAGE = "Welkom bij onze Chatbot!"
DEFAULT_COLOR_CODE = "#0000FF"


# Functie om waarden per tenant op te slaan en te verkrijgen
def get_tenant_redis_key(tenant_id, key):
    """Haal een waarde op uit de Redis cache voor een specifieke tenant."""
    return redis_client.get(f"{tenant_id}:{key}")

def set_tenant_redis_key(tenant_id, key, value, timeout=None):
    """Sla een waarde op in de Redis cache voor een specifieke tenant."""
    namespaced_key = f"{tenant_id}:{key}"
    if timeout:
        redis_client.setex(namespaced_key, timeout, value)
    else:
        redis_client.set(namespaced_key, value)


# Aangepaste initialisatiefunctie voor multi-tenant
def initialize_tenant_defaults(tenant_id):
    # Controleer of de standaardwaarden voor deze tenant al zijn ingesteld
    if not get_tenant_redis_key(tenant_id, 'character_counter'):
        set_tenant_redis_key(tenant_id, 'character_counter', DEFAULT_CHARACTER_COUNTER)
    if not get_tenant_redis_key(tenant_id, 'custom_prompt'):
        set_tenant_redis_key(tenant_id, 'custom_prompt', DEFAULT_PROMPT_TEMPLATE)
    if not get_tenant_redis_key(tenant_id, 'welcome_message'):
        set_tenant_redis_key(tenant_id, 'welcome_message', DEFAULT_WELCOME_MESSAGE)
    if not get_tenant_redis_key(tenant_id, 'title_message'):
        set_tenant_redis_key(tenant_id, 'title_message', DEFAULT_TITLE_MESSAGE)
    if not get_tenant_redis_key(tenant_id, 'color_code'):
        set_tenant_redis_key(tenant_id, 'color_code', DEFAULT_COLOR_CODE)



MONTHLY_LIMIT = 500

def reset_monthly_counter():
    now = datetime.now()
    month_year = now.strftime("%Y-%m")
    for tenant_id in tenants:  # Aanname dat 'tenants' een lijst is van alle tenant ID's
        limit_key = f"{tenant_id}:monthly_limit:{month_year}"
        redis_client.set(limit_key, 0)
        print(f"Reset voltooid voor tenant {tenant_id} voor de maand: {month_year}.")


def check_monthly_limit(tenant_id, month_year):
    limit_key = f"{tenant_id}:monthly_limit:{month_year}"
    current_count = int(redis_client.get(limit_key) or 0)
    return current_count < MONTHLY_LIMIT  # True als limiet niet bereikt is, anders False


def get_current_monthly_usage(tenant_id):
    month_year = datetime.now().strftime("%Y-%m")
    limit_key = f"{tenant_id}:monthly_limit:{month_year}"
    current_count = int(redis_client.get(limit_key) or 0)
    remaining_messages = max(MONTHLY_LIMIT - current_count, 0)
    return current_count, remaining_messages

def get_current_monthly_usage(tenant_id):
    month_year = datetime.now().strftime("%Y-%m")
    limit_key = f"{tenant_id}:monthly_limit:{month_year}"
    current_count = int(redis_client.get(limit_key) or 0)
    remaining_messages = max(MONTHLY_LIMIT - current_count, 0)
    return current_count, remaining_messages



# Reset maandelijkse teller
def reset_monthly_counter():
    now = datetime.now()
    month_year = now.strftime("%Y-%m")
    limit_key = f"monthly_limit:{month_year}"
    redis_client.set(limit_key, 0)
    print(f"Reset voltooid voor maand: {month_year}.")

# Voeg de reset-taak toe aan de scheduler
scheduler = BackgroundScheduler()
scheduler_time = config['scheduler']
scheduler.add_job(reset_monthly_counter, CronTrigger(day=scheduler_time['day'], hour=scheduler_time['hour'], minute=scheduler_time['minute']))
scheduler.start()

print("Scheduler is gestart en taak is toegevoegd.")

    
# Functie om te controleren of de initialisatie al heeft plaatsgevonden
def tenant_is_initialized(tenant_id):
    return redis_client.exists(f"{tenant_id}:character_counter")

# De applicatie initialiseren
def initialize_application():
    tenant_ids = config.get('tenants', [])  # Veronderstelt dat je een lijst van tenants hebt in je config.json
    for tenant_id in tenant_ids:
        if not tenant_is_initialized(tenant_id):
            initialize_redis_for_tenant(tenant_id)
            print(f"Tenant {tenant_id} geïnitialiseerd met standaardwaarden.")
        else:
            print(f"Tenant {tenant_id} was al geïnitialiseerd.")


def initialize_redis_for_tenant(tenant_id):
    # Stel de standaardwaarden in voor een nieuwe tenant
    set_tenant_redis_key(tenant_id, 'character_counter', DEFAULT_CHARACTER_COUNTER)
    set_tenant_redis_key(tenant_id, 'custom_prompt', DEFAULT_PROMPT_TEMPLATE)
    set_tenant_redis_key(tenant_id, 'welcome_message', DEFAULT_WELCOME_MESSAGE)
    set_tenant_redis_key(tenant_id, 'title_message', DEFAULT_TITLE_MESSAGE)
    set_tenant_redis_key(tenant_id, 'color_code', DEFAULT_COLOR_CODE)

    # Log de initialisatie
    logging.info(f"Redis is geïnitialiseerd met standaardwaarden voor tenant {tenant_id}.")

    # Print de waarden in de console om te bevestigen dat ze zijn ingesteld
    print(f"Standaardwaarden ingesteld voor tenant {tenant_id}:")
    print(f" - character_counter: {DEFAULT_CHARACTER_COUNTER}")
    print(f" - custom_prompt: {DEFAULT_PROMPT_TEMPLATE}")
    print(f" - welcome_message: {DEFAULT_WELCOME_MESSAGE}")
    print(f" - title_message: {DEFAULT_TITLE_MESSAGE}")
    print(f" - color_code: {DEFAULT_COLOR_CODE}")

# Roep de initialisatiefunctie aan voor elke tenant
tenants = ['heikant', 'tenant2', 'tenant3', 'tenant4']  # Voorbeeld van tenant ID's
for tenant_id in tenants:
    initialize_tenant_defaults(tenant_id)

def get_tenant_welcome_message(tenant_id):
    logging.info(f"Verkrijgen van welkomstbericht voor tenant {tenant_id}")
    message = get_tenant_redis_key(tenant_id, 'welcome_message')
    if message:
        logging.info(f"Welkomstbericht voor tenant {tenant_id} verkregen uit cache: {message}")
    else:
        message = DEFAULT_WELCOME_MESSAGE
        set_tenant_redis_key(tenant_id, 'welcome_message', message)
        logging.info(f"Standaard welkomstbericht ingesteld en gecached voor tenant {tenant_id}: {message}")
    print(f"Welkomstbericht voor tenant {tenant_id}: {message}")
    return message

def get_tenant_title_message(tenant_id):
    logging.info(f"Verkrijgen van titelbericht voor tenant {tenant_id}")
    message = get_tenant_redis_key(tenant_id, 'title_message')
    if message:
        logging.info(f"Titelbericht voor tenant {tenant_id} verkregen uit cache: {message}")
    else:
        message = DEFAULT_TITLE_MESSAGE
        set_tenant_redis_key(tenant_id, 'title_message', message)
        logging.info(f"Standaard titelbericht ingesteld en gecached voor tenant {tenant_id}: {message}")
    print(f"Titelbericht voor tenant {tenant_id}: {message}")
    return message

def get_tenant_color_settings(tenant_id):
    logging.info(f"Verkrijgen van kleurinstellingen voor tenant {tenant_id}")
    color = get_tenant_redis_key(tenant_id, 'color_code')
    if color:
        logging.info(f"Kleurinstellingen voor tenant {tenant_id} verkregen uit cache: {color}")
    else:
        color = DEFAULT_COLOR_CODE
        set_tenant_redis_key(tenant_id, 'color_code', color)
        logging.info(f"Standaard kleurcode ingesteld en gecached voor tenant {tenant_id}: {color}")
    print(f"Kleurcode voor tenant {tenant_id}: {color}")
    return color

custom_prompt_template = config['prompt_template']


# Functies voor Redis character count management
def increase_character_count(tenant_id, count):
    redis_client.incr(f"{tenant_id}:character_count", count)

def get_character_count(tenant_id):
    count = redis_client.get(f"{tenant_id}:character_count")
    return int(count) if count is not None else 0

def check_limit(tenant_id):
    character_count = get_character_count(tenant_id)
    CHARACTER_LIMIT = 50000  # Voorbeeld limiet
    return character_count >= CHARACTER_LIMIT


# Aanmaken van de PromptTemplate
QA_CHAIN_PROMPT = PromptTemplate.from_template(custom_prompt_template)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0.5)

# Pinecone initialiseren
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name = config['pinecone']['index_name']

# Controleren of index al bestaat, zo niet, aanmaken
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric='cosine', dimension=1536)

# Data laden
urls = []  # Vul hier je URLs in...
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

# Initialisatie van embeddings en index
embed = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
index = pinecone.Index(index_name)

# Data uploaden naar Pinecone
namespace = tenant_id 
batch_limit = 100
texts = []
metadatas = []

for i, record in enumerate(tqdm(data)):  # data moet je zelf definiëren
    metadata = {'wiki-id': str(record['id']), 'source': record['url'], 'title': record['title']}
    record_texts = text_splitter.split_text(record['text'])  # text_splitter moet je zelf definiëren
    record_metadatas = [{"chunk": j, "text": text, **metadata} for j, text in enumerate(record_texts)]
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)

    if len(texts) >= batch_limit:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        try:
            index.upsert(vectors=zip(ids, embeds), namespace=namespace)
            print(f"{len(ids)} records geüpload naar namespace: {namespace}")
        except Exception as e:
            print(f"Upsert mislukt: {e}")
        texts.clear()
        metadatas.clear()


if len(texts) > 0:
    ids = [str(uuid4()) for _ in range(len(texts))]
    embeds = embed.embed_documents(texts)
    try:
        index.upsert(vectors=zip(ids, embeds), namespace=namespace)  # Upsert met de gespecificeerde namespace
        print(f"{len(ids)} records geüpload naar namespace: {namespace}")
    except Exception as e:
        print(f"Upsert mislukt: {e}")

# Vectorstore en LLM instellen
vectorstore = Pinecone(index, embed.embed_query, "text")


# Stel de logger in
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('WelcomeMessageLogger')
handler = logging.FileHandler('welcome_message.log')
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
logger.addHandler(handler)

# App Configuratie
app = Flask(__name__)
CORS(app, support_credentials=True)

app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False  # Sessie eindigt wanneer de browser sluit
app.config['SESSION_USE_SIGNER'] = True  # Zorgt voor extra beveiliging door de cookie te signeren
app.config['SESSION_REDIS'] = redis.StrictRedis.from_url("redis://red-cleho5c15k1s73acpmeg:6379")

# Zorg voor een consistente SECRET_KEY over alle instanties
app.config['SECRET_KEY'] = 'jouw_vaste_geheime_sleutel'  # Vervang dit door jouw geheime sleutel

# Initialiseer Session voor de app
Session(app)

# Flask SQLAlchemy en LoginManager setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://chatbot_78zl_user:LPiX1r6NfpTegHTzWpXcIzt4proyJ7Gq@dpg-clehpets40us73d16ig0-a.frankfurt-postgres.render.com/chatbot_78zl'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# Uw bestaande imports en setup hier
logging.basicConfig(level=logging.INFO)
















# Inlezen van de ongepaste woorden uit het JSON-bestand "scheld.json"
with open("scheld.json", "r") as json_file:
    data = json.load(json_file)

# Haal de ongepaste woorden op en splits ze op komma's
inappropriate_words = data.get("inappropriate_words", "").split(", ")


@app.route('/ask', methods=['POST'])
def ask():
    try:
        request_data = request.get_json()

        # Haal tenant_id op uit de request
        tenant_id = request_data.get('tenant_id')
        if not tenant_id:
            return jsonify({"error": "Tenant ID is vereist"}), 400

        # Controleer de maandelijkse limiet
        month_year = datetime.now().strftime("%Y-%m")
        if not check_monthly_limit(tenant_id, month_year):
            return jsonify({"error": "Maandelijkse limiet bereikt"}), 429

        # Zorg ervoor dat de custom prompt in de cache zit
        ensure_prompt_in_cache(tenant_id)

        # Haal de custom prompt op voor deze tenant
        custom_prompt = get_tenant_redis_key(tenant_id, 'custom_prompt')

        QA_CHAIN_PROMPT = PromptTemplate.from_template(custom_prompt)
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0.5)
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 5, 'namespace': tenant_id}),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True,
        )

        question = request_data.get('question', '')

        # Controleer of de vraag ongepaste woorden bevat
        if any(word in question.lower() for word in inappropriate_words):
            return jsonify({"error": "Ongepaste vraag gedetecteerd"}), 400

        if question.lower() == 'quit':
            return jsonify({"answer": "Chat beëindigd"}), 200

        result = qa_chain({"query": question, "context": ""})

        if result is None or not question:
            return jsonify({"answer": "Geen antwoord ontvangen. Probeer het later opnieuw.", "source_documents": [], "source_links": []}), 200

        # Verhoog de teller alleen als een geldig antwoord is gegenereerd
        redis_client.incr(f"{tenant_id}:monthly_limit:{month_year}")

        answer = result.get('result', 'Sorry, ik kan deze vraag niet beantwoorden.')
        source_documents = result.get('source_documents', [])
        source_documents_serializable = [vars(doc) for doc in source_documents]

        source_links = [{"url": doc.get('source', ''), "title": doc.get('title', '')} for doc in source_documents_serializable]

        return jsonify({"answer": answer, "source_documents": source_documents_serializable, "source_links": source_links}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


                
@app.route('/<tenant_id>/get_custom_prompt', methods=['GET'])
def get_custom_prompt(tenant_id):
    """Haal de aangepaste prompt uit Redis voor de tenant."""
    prompt = get_tenant_redis_key(tenant_id, 'custom_prompt')
    if not prompt:
        prompt = DEFAULT_PROMPT_TEMPLATE
        set_tenant_redis_key(tenant_id, 'custom_prompt', prompt)  # Cache de standaard prompt in Redis voor de tenant

    return jsonify({"custom_prompt": prompt}), 200

def ensure_prompt_in_cache(tenant_id):
    prompt = get_tenant_redis_key(tenant_id, 'custom_prompt')
    if prompt is None:
        # Als de prompt niet in de cache is, gebruik de standaardwaarde voor de betreffende tenant.
        prompt_to_cache = DEFAULT_PROMPT_TEMPLATE
        set_tenant_redis_key(tenant_id, 'custom_prompt', prompt_to_cache)

@app.before_first_request
def initialize():
    logging.info("Eerste verzoek ontvangen; initialiseren van de cache voor alle tenants")
    tenants = ['tenant1', 'tenant2', 'tenant3', 'heikant']  # Lijst van tenants
    for tenant_id in tenants:
        # Initialiseer de waarden voor elke tenant
        initialize_tenant_defaults(tenant_id)

        # Log de waarden voor elke tenant
        custom_prompt = get_tenant_redis_key(tenant_id, 'custom_prompt')
        welcome_message = get_tenant_redis_key(tenant_id, 'welcome_message')
        title_message = get_tenant_redis_key(tenant_id, 'title_message')
        color_code = get_tenant_redis_key(tenant_id, 'color_code')

        # Log de opgehaalde of ingestelde waarden
        logging.info(f"Cache geïnitialiseerd voor tenant {tenant_id}, custom_prompt is '{custom_prompt}'")
        logging.info(f"Cache geïnitialiseerd voor tenant {tenant_id}, welcome_message is '{welcome_message}'")
        logging.info(f"Cache geïnitialiseerd voor tenant {tenant_id}, title_message is '{title_message}'")
        logging.info(f"Cache geïnitialiseerd voor tenant {tenant_id}, color_code is '{color_code}'")



@app.route('/<tenant_id>/get_welcome_message', methods=['GET'])
def get_welcome_message(tenant_id):
    """Haal het welkomstbericht uit Redis voor de tenant."""
    message = get_tenant_welcome_message(tenant_id)
    if not message:
        message = DEFAULT_WELCOME_MESSAGE
        set_tenant_redis_key(tenant_id, 'welcome_message', message)  # Cache het standaard welkomstbericht in Redis voor de tenant

    return jsonify({"welcome_message": message}), 200

@app.route('/<tenant_id>/get_title_message', methods=['GET'])
def api_get_title_message(tenant_id):
    """Haal het titelbericht uit Redis voor de tenant."""
    logging.info(f"GET request received for /{tenant_id}/get_title_message")
    title_message = get_tenant_title_message(tenant_id)

    if title_message:
        # Als er een titelboodschap gevonden is, sturen we deze terug
        return jsonify({'title_message': title_message})
    else:
        # Als er geen bericht is gevonden of als er een fout is opgetreden, sturen we een passende foutmelding
        return jsonify({'error': 'Geen titelbericht gevonden voor de opgegeven tenant'}), 404

@app.route('/<tenant_id>/get_color', methods=['GET'])
def api_get_color(tenant_id):
    """Haal de kleurinstellingen uit Redis voor de tenant."""
    logging.info(f"GET request received for /{tenant_id}/get_color")
    color = get_tenant_color_settings(tenant_id)

    if color:
        # Als er kleurinstellingen gevonden zijn, log dit dan en stuur de kleur terug
        logging.info(f"Kleurinstellingen voor tenant {tenant_id} verkregen: {color}")
        return jsonify({'color': color})
    else:
        # Als er geen kleurinstellingen gevonden zijn, log een foutmelding en stuur een foutmelding terug
        error_message = 'Geen kleurinstellingen gevonden voor de opgegeven tenant'
        logging.error(f"{error_message} voor tenant {tenant_id}")
        return jsonify({'error': error_message}), 404



def initialize_application():
    tenant_ids = config.get('tenants', [])  # Veronderstelt dat je een lijst van tenants hebt in je config.json
    for tenant_id in tenant_ids:
        if not tenant_is_initialized(tenant_id):
            initialize_tenant_defaults(tenant_id)  # Deze functie initialiseert de tenant waarden
            print(f"Tenant {tenant_id} geïnitialiseerd met standaardwaarden.")
        else:
            print(f"Tenant {tenant_id} was al geïnitialiseerd.")



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # Stel de logging in
    initialize_application()  # Initialiseer de applicatie met tenant waarden
    app.run(debug=True, threaded=True)  # Start de Flask-applicatie

































































