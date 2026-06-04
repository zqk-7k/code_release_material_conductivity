import os
import json
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# =========================
# 0. Configuration & initialization
# =========================
load_dotenv()

# Database config
user = os.getenv("DB_USER", "root")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST", "127.0.0.1")
port = int(os.getenv("DB_PORT", "3306"))
database = os.getenv("DB_NAME")

DB_CONNECTION_STR = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

# API config (recommended: read from environment variables)
API_KEY = os.getenv("OPENAI_API_KEY")
model_name=os.getenv("MODEL_NAME")

# LLM initialization
llm = ChatOpenAI(
    api_key=API_KEY,
    model=model_name, # Recommended: use a stable model version
    temperature=0
)

# Prompt definition
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional materials-science translator.\n"
     "Translate items to English.\n"
     "- Keep technical meaning.\n"
     "- If an item is already English, keep it unchanged.\n"
     "- Return ONLY a valid JSON array of strings, same length and same order as the input array.\n"),
    ("human", "Input JSON array:\n{items_json}")
])

chain = prompt | llm | JsonOutputParser()

# =========================
# 1. Helper functions (reuse your logic)
# =========================
def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def translate_unique(values, chunk_size=50, max_concurrency=5):
    """
    Extract unique values -> batch translate -> return a dict mapping
    """
    # Filter non-strings and empty values
    values = [v for v in values if isinstance(v, str) and v.strip()]
    if not values:
        return {}

    print(f"Translating {len(values)} unique terms...")

    chunks = list(chunked(values, chunk_size))
    inputs = [{"items_json": json.dumps(c, ensure_ascii=False)} for c in chunks]

    # Batch calls with concurrency
    try:
        outputs = chain.batch(inputs, config={"max_concurrency": max_concurrency})
    except Exception as e:
        print(f"Batch translation failed: {e}")
        return {}

    translated = []
    for out in outputs:
        if isinstance(out, list):
            translated.extend(out)
        else:
            # Fallback: if the model doesn't return a list
            translated.extend(["Error"] * chunk_size)

    # Length check
    if len(translated) != len(values):
        print(f"Warning: Length mismatch {len(values)} vs {len(translated)}. Truncating/Padding.")
        # Simple alignment to avoid errors
        min_len = min(len(values), len(translated))
        values = values[:min_len]
        translated = translated[:min_len]

    return dict(zip(values, translated))

# =========================
# 2. Main flow
# =========================
def main():
    # 1. Connect to database
    print(f"Connecting to database: {database}...")
    engine = create_engine(DB_CONNECTION_STR)

    # 2. Read fields to translate from the raw table (id used to align updates later)
    # We need the raw Chinese text and the id
    read_sql = """
               SELECT sample_id, synthesis_method, processing_route
               FROM raw_conductivity_samples \
               """
    print("Reading raw data...")
    df = pd.read_sql(read_sql, engine)
    print(f"Loaded {len(df)} rows.")

    # 3. Extract unique values and translate (synthesis_method)
    print("\n--- Processing Synthesis Methods ---")
    sm_unique = df["synthesis_method"].dropna().unique().tolist()
    sm_map = translate_unique(sm_unique, chunk_size=50, max_concurrency=3)

    # 4. Extract unique values and translate (processing_route)
    print("\n--- Processing Processing Routes ---")
    pr_unique = df["processing_route"].dropna().unique().tolist()
    pr_map = translate_unique(pr_unique, chunk_size=50, max_concurrency=3)

    # 5. Map back to the DataFrame
    # Note: we build data used for the UPDATE
    # If map can't find a key it becomes NaN; convert to None so SQL sees NULL
    df["synthesis_method_en"] = df["synthesis_method"].map(sm_map).replace({pd.NA: None, float('nan'): None})
    df["processing_route_en"] = df["processing_route"].map(pr_map).replace({pd.NA: None, float('nan'): None})

    # 6. Prepare batch update data
    # Build a list of parameter dicts
    update_data = []
    for _, row in df.iterrows():
        # Only update when at least one field has a value
        if row["synthesis_method_en"] or row["processing_route_en"]:
            update_data.append({
                "s_en": row["synthesis_method_en"],
                "p_en": row["processing_route_en"],
                "sid": row["sample_id"]
            })

    print(f"\nPreparing to update {len(update_data)} rows in 'tmp_translate_result'...")

    # 7. Execute batch UPDATE
    if update_data:
        # Define SQL statement (uses bound params :name)
        update_stmt = text("""
                           UPDATE tmp_translate_result
                           SET synthesis_method = :s_en,
                               processing_route = :p_en
                           WHERE sample_id = :sid
                           """)

        with engine.begin() as conn:  # begin() manages the transaction automatically
            # SQLAlchemy optimizes executing a list into executemany
            # Run in batches to avoid Packet Too Large errors (1000 per batch)
            batch_size = 1000
            for i in tqdm(range(0, len(update_data), batch_size), desc="Updating DB"):
                batch = update_data[i : i + batch_size]
                conn.execute(update_stmt, batch)

        print("Database update completed successfully.")
    else:
        print("No data to update.")

if __name__ == "__main__":
    from tqdm import tqdm  # progress bar
    main()
