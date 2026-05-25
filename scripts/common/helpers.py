import re
import requests
import csv
import random
from SPARQLWrapper import SPARQLWrapper, JSON

random.seed(42)

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
PAGEVIEW_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
USER_AGENT = "CANDOR-benchmark-research/1.0 (research project)"


def get_sparql_client():
    client = SPARQLWrapper(SPARQL_ENDPOINT)
    client.addCustomHttpHeader("User-Agent", USER_AGENT)
    client.setReturnFormat(JSON)
    return client


def is_valid_label(text):
    if not text or not str(text).strip():
        return False
    if re.fullmatch(r"Q\d+", str(text).strip()):
        return False
    if re.search(r"\bQ\d{4,}\b", str(text)):
        return False
    if len(str(text).strip()) < 2:
        return False
    return True


def deep_clean(facts, keys):
    cleaned = []
    seen = set()
    for f in facts:
        checks = [is_valid_label(f.get(k, "")) for k in keys]
        key = tuple(f.get(k, "") for k in keys)
        if all(checks) and key not in seen:
            cleaned.append(f)
            seen.add(key)
    return cleaned


def save_csv(rows, path, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_monthly_pageviews(article_title, year=2023):
    url = (f"{PAGEVIEW_BASE}/en.wikipedia/all-access/user/"
           f"{article_title}/monthly/{year}010100/{year}123100")
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=20)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    data = r.json()
    items = data.get("items", [])
    if not items:
        return 0
    return round(sum(i.get("views", 0) for i in items) / len(items))


def get_entity_wikipedia_title(entity_name):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": entity_name,
        "format": "json",
        "srlimit": 1,
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    results = r.json().get("query", {}).get("search", [])
    if not results:
        return None
    return results[0]["title"].replace(" ", "_")


def check_entity_salience(entity_name, threshold=50000, year=2023):
    title = get_entity_wikipedia_title(entity_name)
    if not title:
        return False
    views = get_monthly_pageviews(title, year=year)
    if views is None:
        return False
    return views >= threshold


RELATION_TEMPLATES = {
    "P19":  "What is the birthplace of {entity}?",
    "P106": "What is the occupation of {entity}?",
    "P27":  "What is the nationality of {entity}?",
    "P39":  "What position did {entity} hold?",
    "P569": "When was {entity} born?",
    "P571": "When was {entity} founded?",
    "P17":  "Which country is {entity} located in?",
    "P175": "Who performed {entity}?",
    "P57":  "Who directed {entity}?",
    "P50":  "Who is the author of {entity}?",
}


def triple_to_question(entity_label, relation_pid):
    template = RELATION_TEMPLATES.get(relation_pid)
    if not template:
        return None
    return template.format(entity=entity_label)
