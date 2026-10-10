import re
import emoji

SLANG_MAP = {
    "idk": "i do not know",
    "tbh": "to be honest",
    "fml": "i am devastated",
    "smh": "disappointed",
    "ngl": "not going to lie",
    "pls": "please",
    "plz": "please",
    "im": "i am",
    "dont": "do not",
    "cant": "cannot",
    "wont": "will not",
    "gonna": "going to",
    "wanna": "want to"
}

def clean_and_normalize(text: str) -> str:
    # 1. Expand emojis into natural descriptive words
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.replace("_", " ").replace(":", " ")

    # 2. Lowercase and expand slang acronyms
    tokens = text.lower().split()
    expanded = [SLANG_MAP.get(tok, tok) for tok in tokens]
    text = " ".join(expanded)

    # 3. Strip non-alphanumeric characters (keep basic spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
