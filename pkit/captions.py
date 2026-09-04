"""Slide captions authored in markdown.

Deck scripts pull their titles/subtitles from an .md file the author
edits directly (rgb_reports/slide_captions/*.md), so the words on the
slides are owned prose, not generator string literals.

File format — one section per slide key:

    # SELF
    **The title line (bold, may wrap)**

    Subtitle paragraph(s). Multiple paragraphs are joined with a space
    before rendering. Placeholders like {n_self} are substituted from
    the context the deck script passes; unknown braces ({X}) pass
    through untouched.
"""
import re


def load(path):
    """Parse a captions file -> {key: {"title": str, "sub": str}}."""
    text = open(path).read()
    out = {}
    for chunk in re.split(r"^# +", text, flags=re.M)[1:]:
        lines = chunk.splitlines()
        key = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        m = re.match(r"\*\*(.+?)\*\*\s*", body, re.S)
        if not m:
            raise ValueError(f"[captions] section {key!r} has no **title**")
        title = " ".join(m.group(1).split())
        rest = body[m.end():].strip()
        paras = [" ".join(q.split()) for q in re.split(r"\n\s*\n", rest) if q.strip()]
        out[key] = {"title": title, "sub": " ".join(paras), "paras": paras}
    return out


def render(s, ctx):
    """Substitute known {placeholders}; unknown braces pass through."""
    return re.sub(r"\{(\w+)\}",
                  lambda m: str(ctx[m.group(1)]) if m.group(1) in ctx
                  else m.group(0), s)
