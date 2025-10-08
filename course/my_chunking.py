import re
import os
from tqdm.auto import tqdm
from dotenv import load_dotenv
from my_ingestion import read_repo_data

load_dotenv()

# for intelligent chunking
try:
    from groq import Groq
except ImportError:
    Groq = None

# 1. Simple sliding-window chunking
def sliding_window(seq, size=2000, step=1000):
    """
    Split text into overlapping chunks.
    Example:
        size=2000, step=1000 → overlap of 1000 characters.
    """
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        chunk = seq[i:i + size]
        result.append({'start': i, 'chunk': chunk})
        if i + size >= n:
            break
    return result


def apply_sliding_window(docs, size=2000, step=1000):
    """Apply sliding-window chunking to all documents."""
    all_chunks = []
    for doc in tqdm(docs, desc="Sliding Window Chunking"):
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content')
        chunks = sliding_window(doc_content, size, step)
        for chunk in chunks:
            chunk.update(doc_copy)
        all_chunks.extend(chunks)
    return all_chunks

# 2. Section based splitting
def split_markdown_by_level(text, level=2):
    """
    Split markdown text into sections based on header levels.
    Args:
        text: Markdown text
        level: Header level to split on (e.g., 2 for '##')
    Returns:
        List of sections with their start positions
    """
    # This regex matches markdown headers of the specified level
    # For level 2, it matches lines starting with '## '
    header_pattern = r'^(#{' + str(level) + r'} )(.+)$'
    pattern = re.compile(header_pattern, re.MULTILINE)

    # Split and keep the headers
    parts = pattern.split(text)

    sections = []
    for i in range(1, len(parts), 3):
        # We keep by 3 because regex.split() with capturing groups returns:
        # [before_match, group1, group2, after_match, ...]
        # here group1 is "## ", group2 is the header text
        header = parts[i] + parts[i + 1] # "## " + "Title"
        header = header.strip()

        # Get the content after this header
        content = ""
        if i+2 < len(parts):
            content = parts[i+2].strip()

        if content:
            section = f'{header}\n\n{content}'
        else:
            section = header
        sections.append(section)
    return sections

def apply_section_split(docs, level=2):
    """Apply section-based splitting to markdown documents."""
    all_chunks = []
    for doc in tqdm(docs, desc=f"Splitting by Markdown Level {level}"):
        doc_copy = doc.copy()
        doc_content = doc_copy.pop("content")
        sections = split_markdown_by_level(doc_content, level)
        for section in sections:
            section_doc = doc_copy.copy()
            section_doc["section"] = section
            all_chunks.append(section_doc)
    return all_chunks

# 3. Intelligent chunking using LLM (Groq)
prompt_template = """
Split the provided document into logical sections that make sense for a Q&A system.

Each section should be self-contained and cover a specific topic or concept.

<DOCUMENT>
{document}
</DOCUMENT>

Use this format:

## Section Name

Section content with all relevant details

---

## Another Section Name

Another section content

---
""".strip()

def intelligent_chunking(text, client, model="llama-3.1-8b-instant", max_tokens_in=5500):
    """Use Groq's LLM to intelligently chunk a document."""
    # prevent oversize requests
    if len(text) > max_tokens_in * 4:  # rough 4 chars ≈ 1 token
        # split in half recursively
        mid = len(text) // 2
        return (intelligent_chunking(text[:mid], client, model, max_tokens_in)
                + intelligent_chunking(text[mid:], client, model, max_tokens_in))
    prompt = prompt_template.format(document=text)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=4096,
    )
    result_text = completion.choices[0].message.content
    sections = [s.strip() for s in result_text.split("---") if s.strip()]
    return sections
    
def apply_intelligent_chunking(docs, model="llama-3.1-8b-instant"):
    """Apply Groq-based intelligent chunking to all documents."""
    if Groq is None:
        raise ImportError("groq not installed. Run: uv add groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ GROQ_API_KEY not found in .env file")

    client = Groq(api_key=api_key)
    all_chunks = []

    for doc in tqdm(docs, desc="Groq Intelligent Chunking"):
        doc_copy = doc.copy()
        doc_content = doc_copy.pop("content")
        try:
            sections = intelligent_chunking(doc_content, client, model)
            for section in sections:
                section_doc = doc_copy.copy()
                section_doc["section"] = section
                all_chunks.append(section_doc)
        except Exception as e:
            print(f"⚠️ Error processing {doc_copy.get('title', 'Unknown')}: {e}")
            continue
    return all_chunks

# if __name__ == "__main__":
#     print("📥 Loading EvidentlyAI repository...")
#     evidently_docs = read_repo_data("evidentlyai", "docs")
#     print(f"✅ Loaded {len(evidently_docs)} documents.")

#     # --- Method 1: Sliding Window ---
#     evidently_chunks_window = apply_sliding_window(evidently_docs)
#     print(f"✅ Created {len(evidently_chunks_window)} sliding-window chunks.")

#     # --- Method 2: Section-Based ---
#     evidently_chunks_section = apply_section_split(evidently_docs)
#     print(f"✅ Created {len(evidently_chunks_section)} section-based chunks.")

#     # --- Method 3: Groq Intelligent Chunking ---
#     evidently_chunks_groq = apply_intelligent_chunking(evidently_docs)
#     print(f"✅ Created {len(evidently_chunks_groq)} Groq intelligent chunks.")

#     # --- Inspect a sample ---
#     print("\n🔍 Example chunk:")
#     print(evidently_chunks_window[0]["chunk"][:500])

#     print("\n🎯 Day 2 complete! Compare all three methods and pick the best for your project.")