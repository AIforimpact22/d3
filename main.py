import os
import re
import json
from flask import Flask, request, jsonify, render_template
from openai import OpenAI

# --- Config (as requested) ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # optional (only for AI mode)
PORT = int(os.getenv("PORT", "5000"))

app = Flask(__name__)

# Optional OpenAI client (AI mode)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Brand (bottom-right watermark on the chart)
BRAND_LOGO_URL = "https://i.imgur.com/STm5VaG.png"
BRAND_NAME = "Ai For Impact"
BRAND_SITE = "www.aiforimpact.net"


@app.route("/")
def index():
    return render_template(
        "base.html",
        brand_logo_url=BRAND_LOGO_URL,
        brand_name=BRAND_NAME,
        brand_site=BRAND_SITE,
        model_name=OPENAI_MODEL,
        ai_available=bool(client)
    )

# ----------------- Local prompt → nodes[] (no LLM) -----------------

STOPWORDS = set("""
a an and the to for of with into from by our your their his her its on in over under at as is are be been were was do does did
then next after before finally also so that which who whom whose this those these it they we you i
system app module service ai agent process workflow task stage phase item node data info information
""".split())

GROUP_KEYWORDS = [
    ("input",       r"(ingest|capture|record|collect|scan|load|update|read|write|sync|import|export|fetch)"),
    ("analysis",    r"(analy|calculat|compute|score|cluster|classif|detect|match|rank|measure|metric|velocity|average|trend)"),
    ("model",       r"(train|fit|learn|model|regress|classif|predict|inference|forecast)"),
    ("po",          r"(po|purchase|order|reorder|supplier|vendor|procure|quote|tender)"),
    ("validation",  r"(validate|approve|review|audit|monitor|observe|control|budget|policy|sign.?off|qa|qc|verify)"),
    ("communication", r"(notify|message|email|sms|webhook|post|publish|send|alert|broadcast)"),
    ("ops",         r"(deploy|operate|scale|backup|restore|schedule|cron|retry|queue|cache|index)"),
]

def _sentences(prompt: str):
    parts = re.split(r"(?:\n+|[.;:]+|\bthen\b|\band then\b|\bnext\b|\bafter\b|→|=>|->)", prompt, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p and p.strip()]

def _clean_words(s: str):
    """
    Keep alnum/underscore tokens, drop stopwords, and IGNORE numeric-only tokens.
    Ensures every token has at least one alphabetic character.
    """
    raw = re.split(r"[^A-Za-z0-9_]+", s.lower())
    return [w for w in raw if w and (w not in STOPWORDS) and re.search(r"[a-z]", w)]

def _pascal_from_words(words, cap=6):
    if not words: return "Step"
    picked = words[:cap]
    pascal = "".join(w[:1].upper() + w[1:] for w in picked)
    if not re.match(r"^[A-Za-z]", pascal):  # extra guard
        pascal = "Step" + pascal
    return pascal

def _guess_group(text: str):
    for group, pat in GROUP_KEYWORDS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return group
    return "flow"

def _stable_size(name: str):
    s = sum(ord(c) for c in name)
    return 350 + (s % 551)  # 350..900

def _extract_arrow_edges(prompt: str):
    pairs = []
    for m in re.finditer(r"([A-Za-z0-9 _\-\/]{2,}?)\s*(?:->|→|=>)\s*([A-Za-z0-9 _\-\/]{2,})", prompt):
        a = m.group(1).strip(); b = m.group(2).strip()
        if a and b: pairs.append((a, b))
    return pairs

def _local_nodes_from_prompt(prompt: str):
    sents = _sentences(prompt)
    if len(sents) < 3:
        extra = re.split(r"\band\b|,|\u2013|\u2014", prompt, flags=re.IGNORECASE)
        sents = [x.strip() for x in extra if x.strip()]

    nodes, seen = [], set()
    for s in sents:
        words = _clean_words(s)
        # skip sentences that yield no alphabetic tokens
        if not words:
            continue
        step = _pascal_from_words(words, cap=6)
        # avoid generic "Step" (can happen if words filtered out)
        if step.lower() == "step":
            continue
        group = _guess_group(s)
        dotted = f"method.{group}.{step}"
        if dotted in seen:
            continue
        seen.add(dotted)
        nodes.append({"name": dotted, "size": _stable_size(dotted), "imports": []})

    # NOTE: Removed numeric-driven padding to prevent extras like Step350/Step500/etc.
    # We do NOT fabricate extra nodes anymore. We only connect what the prompt produced.

    # Arrow hints A -> B : edge from A to B
    arrow_pairs = _extract_arrow_edges(prompt)
    if arrow_pairs and nodes:
        def fuzzy_find(txt):
            tw = _clean_words(txt)
            if not tw: return None
            needle = re.escape(tw[0])
            for n in nodes:
                if re.search(needle, n["name"], re.IGNORECASE):
                    return n["name"]
            return None
        for a_txt, b_txt in arrow_pairs:
            a = fuzzy_find(a_txt); b = fuzzy_find(b_txt)
            if a and b and a != b:
                for n in nodes:
                    if n["name"] == a and b not in n["imports"]:
                        n["imports"].append(b)

    # Ensure at least some edges for visible curves when possible
    if sum(len(n["imports"]) for n in nodes) == 0 and len(nodes) >= 2:
        for i in range(1, len(nodes)):
            prev = nodes[i - 1]["name"]
            if prev not in nodes[i]["imports"]:
                nodes[i]["imports"].append(prev)

    # Scrub to valid dotted names and non-dangling imports
    dotted_ok = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+$")
    names = {n["name"] for n in nodes if dotted_ok.match(n["name"])}
    out = []
    for n in nodes:
        if n["name"] not in names:
            continue
        n["imports"] = [i for i in dict.fromkeys(n.get("imports", [])) if i in names and i != n["name"]]
        out.append(n)
    return out

# ----------------- Optional AI path -----------------

def _extract_nodes_array(text: str):
    if not text: return None
    try:
        obj = json.loads(text)
        if isinstance(obj, list): return obj
        if isinstance(obj, dict) and isinstance(obj.get("nodes"), list): return obj["nodes"]
    except Exception: pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, list): return obj
            if isinstance(obj, dict) and isinstance(obj.get("nodes"), list): return obj["nodes"]
        except Exception: pass
    arrays = re.findall(r"\[[\s\S]*\]", text)
    arrays.sort(key=len, reverse=True)
    for arr in arrays:
        try:
            obj = json.loads(arr)
            if isinstance(obj, list): return obj
        except Exception:
            continue
    return None

def _coerce_and_validate_nodes(nodes):
    if not isinstance(nodes, list): return []
    dotted_ok = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+$")
    out, seen = [], set()
    for n in nodes:
        if not isinstance(n, dict): continue
        name = n.get("name"); size = n.get("size", 600); imports = n.get("imports", [])
        if not isinstance(name, str) or not dotted_ok.match(name): continue
        try: size = int(size)
        except Exception: size = 600
        if not isinstance(imports, list): imports = []
        imports = [i for i in imports if isinstance(i, str) and dotted_ok.match(i)]
        if name in seen: continue
        seen.add(name)
        out.append({"name": name, "size": max(1, min(100000, size)), "imports": imports})
    names = {n["name"] for n in out}
    for n in out:
        n["imports"] = [i for i in n["imports"] if i in names and i != n["name"]]
    # Keep the "chain" fallback only if there are at least 2 nodes and no imports
    if sum(len(n["imports"]) for n in out) == 0 and len(out) >= 2:
        for i in range(1, len(out)):
            prev = out[i - 1]["name"]
            if prev not in out[i]["imports"]:
                out[i]["imports"].append(prev)
    return out

def _ai_nodes_from_prompt(prompt: str):
    if not client:
        return _local_nodes_from_prompt(prompt)

    system_msg = (
        "You are a workflow graph generator. Output ONLY a JSON array of nodes. "
        "Each node is {\"name\": string, \"size\": integer, \"imports\": string[]}. "
        "• Flat array of 8–24 leaf nodes, names are dotted paths with 3+ segments. "
        "• imports reference other names in the SAME array. No prose."
    )
    user_msg = (
        f"Description:\n{prompt}\n\n"
        "Return ONLY the JSON array like:\n"
        "[{\"name\":\"method.input.PrepareData\",\"size\":800,\"imports\":[]}, ...]"
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            # Do not set 'temperature' (some models only accept default)
            max_completion_tokens=2048,
        )
        content = (resp.choices[0].message.content or "").strip()
        nodes = _extract_nodes_array(content) or _local_nodes_from_prompt(prompt)
        nodes = _coerce_and_validate_nodes(nodes)
        if len(nodes) < 3:
            return _local_nodes_from_prompt(prompt)
        return nodes
    except Exception:
        return _local_nodes_from_prompt(prompt)

# ----------------- API -----------------

@app.post("/api/generate")
def api_generate():
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    mode = (body.get("mode") or "local").lower()  # "local" (default) or "ai"

    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    nodes = _ai_nodes_from_prompt(prompt) if mode == "ai" else _local_nodes_from_prompt(prompt)
    if len(nodes) < 3:
        return jsonify({"error": "Could not derive enough steps from the prompt. Add more detail."}), 400

    return jsonify({"nodes": nodes})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=PORT)
