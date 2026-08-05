"""Hand-authored, verified GOLD examples that teach the model to use SAFE,
non-controversial PLACEHOLDER-IMAGE sources instead of web-searching for images
(the v6/v8 field-test bug: it reached for the dead via.placeholder.com or, worse,
tried to *search* for images and hallucinated URLs).

Two behaviors are taught:
  1. "build an HTML page with images" -> use a live placeholder host
     (placehold.co / dummyimage.com / picsum.photos seeded) or an inline SVG
     data-URI (fully self-contained, zero network, zero controversy risk).
  2. "find/search for images and put them in" -> DO NOT search; deliver the page
     with placeholder images and note they can be swapped for real ones.

Hosts here were curl-verified live (200). Gold, not teacher-generated, precisely
so no dead/uncontrolled URL sneaks in. Every solution is checked with
datalib.is_valid_html before it's written. Output: image_gold.jsonl (kind=web),
picked up by mix.py and fence-normalized like the rest of the web data.
"""
import json
from datalib import is_valid_html, is_multiline_formatted

OUT = "image_gold.jsonl"

# A reusable inline-SVG placeholder (self-contained; the safest option).
def svg(label, w=600, h=400, bg="223", fg="fff"):
    return (f"data:image/svg+xml;utf8,"
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>"
            f"<rect width='100%25' height='100%25' fill='%23{bg}'/>"
            f"<text x='50%25' y='50%25' fill='%23{fg}' font-size='28' "
            f"text-anchor='middle' dominant-baseline='middle'>{label}</text></svg>")


EXAMPLES = [
    # --- placehold.co: labeled boxes ---------------------------------------
    {"q": "Build an HTML fragment: a gallery of three product cards, each with a "
          "placeholder image, a title, and a price. Use placeholder images.",
     "a": """<div class="gallery">
  <div class="card">
    <img src="https://placehold.co/300x200?text=Product+A" alt="Product A" width="300" height="200">
    <h3>Product A</h3>
    <p>$19.99</p>
  </div>
  <div class="card">
    <img src="https://placehold.co/300x200?text=Product+B" alt="Product B" width="300" height="200">
    <h3>Product B</h3>
    <p>$24.99</p>
  </div>
  <div class="card">
    <img src="https://placehold.co/300x200?text=Product+C" alt="Product C" width="300" height="200">
    <h3>Product C</h3>
    <p>$29.99</p>
  </div>
</div>"""},

    # --- explicit restraint: user asks to "find images" --------------------
    {"q": "Find some real images online for a hero banner about hiking and put "
          "them in an HTML section.",
     "a": """I can't fetch live images, so I've used placeholder images you can swap for real ones later (just replace each src). Here's the section:

<section class="hero">
  <img src="https://placehold.co/1200x400?text=Hiking+Hero" alt="Scenic hiking trail" width="1200" height="400">
  <h1>Explore the Trails</h1>
  <p>Placeholder image above — replace the src with your own photo when ready.</p>
</section>"""},

    # --- picsum, seeded (deterministic realistic photos) -------------------
    {"q": "Create an HTML fragment showing a blog post preview with a realistic "
          "placeholder photo, a headline, and a one-line excerpt.",
     "a": """<article class="preview">
  <img src="https://picsum.photos/seed/blogpost/600/300" alt="Article thumbnail" width="600" height="300">
  <h2>How We Cut Build Times in Half</h2>
  <p>A short look at the caching change that made the difference.</p>
</article>"""},

    # --- inline SVG: fully self-contained ----------------------------------
    {"q": "Build a self-contained HTML fragment with an image placeholder that "
          "works offline (no external requests).",
     "a": f"""<figure class="offline-card">
  <img src="{svg('No External Requests')}" alt="Offline placeholder" width="600" height="400">
  <figcaption>This placeholder is an inline SVG — no network needed.</figcaption>
</figure>"""},

    # --- dummyimage.com, colored ------------------------------------------
    {"q": "Give me an HTML avatar row: four circular placeholder avatars for a "
          "team page.",
     "a": """<div class="team">
  <img class="avatar" src="https://dummyimage.com/80x80/4a90d9/ffffff&text=AB" alt="Team member AB" width="80" height="80">
  <img class="avatar" src="https://dummyimage.com/80x80/50b96a/ffffff&text=CD" alt="Team member CD" width="80" height="80">
  <img class="avatar" src="https://dummyimage.com/80x80/d95c4a/ffffff&text=EF" alt="Team member EF" width="80" height="80">
  <img class="avatar" src="https://dummyimage.com/80x80/9b59b6/ffffff&text=GH" alt="Team member GH" width="80" height="80">
</div>"""},

    # --- full page with labeled step images (the launch-profile shape) -----
    {"q": "Build a complete, self-contained HTML page that explains a 3-step "
          "process, each step illustrated with a labeled placeholder image.",
     "a": """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>The Process</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; }
    .step { display: flex; gap: 1rem; align-items: center; margin: 1.5rem 0; }
    .step img { border-radius: 8px; }
  </style>
</head>
<body>
  <h1>The Process</h1>
  <section class="step">
    <img src="https://placehold.co/200x140?text=Step+1" alt="Step 1: Prepare" width="200" height="140">
    <p><strong>Step 1 — Prepare.</strong> Gather everything you need before you start.</p>
  </section>
  <section class="step">
    <img src="https://placehold.co/200x140?text=Step+2" alt="Step 2: Build" width="200" height="140">
    <p><strong>Step 2 — Build.</strong> Assemble the pieces in order.</p>
  </section>
  <section class="step">
    <img src="https://placehold.co/200x140?text=Step+3" alt="Step 3: Review" width="200" height="140">
    <p><strong>Step 3 — Review.</strong> Check the result and refine.</p>
  </section>
</body>
</html>"""},

    # --- restraint w/ a search tool conceptually available -----------------
    {"q": "Search the web for pictures of golden retrievers and build a small "
          "HTML card with one.",
     "a": """Rather than searching (image search results aren't reliable to hotlink), I've used a placeholder you can replace with a real photo:

<div class="pet-card">
  <img src="https://placehold.co/400x300?text=Golden+Retriever" alt="Golden retriever" width="400" height="300">
  <h3>Golden Retriever</h3>
  <p>Swap the image src for your own photo when you have one.</p>
</div>"""},
]


def main():
    kept = 0
    with open(OUT, "w") as f:
        for ex in EXAMPLES:
            sol = ex["a"]
            # verify the markup is structurally sound + formatted (dogfood the gate)
            full = sol.lstrip().lower().startswith("<!doctype")
            html_part = sol  # restraint answers have a prose lead + markup; lint whole
            if not is_valid_html(html_part, allow_fragment=not full):
                print("SKIP (invalid HTML):", ex["q"][:60]); continue
            if not is_multiline_formatted(html_part):
                print("SKIP (minified):", ex["q"][:60]); continue
            row = {"messages": [{"role": "user", "content": ex["q"]},
                                {"role": "assistant", "content": sol}],
                   "kind": "web", "lang": "HTML", "img": True}
            f.write(json.dumps(row) + "\n"); kept += 1
    print(f"wrote {kept}/{len(EXAMPLES)} verified image-gold examples -> {OUT}")


if __name__ == "__main__":
    main()
