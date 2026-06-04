import os
import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine
from dotenv import load_dotenv  # Load environment variables from .env
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# =========================
# 0. Load configuration
# =========================
# Load the .env file
load_dotenv()

# Read OpenAI API key
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please configure OPENAI_API_KEY in your .env file.")
model_name=os.getenv("MODEL_NAME")

# Read database config (from your provided code)
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST", "127.0.0.1")
port = int(os.getenv("DB_PORT", "3306"))
database = os.getenv("DB_NAME")

# Build SQLAlchemy connection string
# Format: mysql+pymysql://user:password@host:port/database
DB_CONNECTION_STR = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

# =========================
# 1. Connect to DB & read data
# =========================
print(f"Connecting to database '{database}' at {host}...")
engine = create_engine(DB_CONNECTION_STR)

# Read source data
read_sql = """
           SELECT sample_id, material_source_and_purity
           FROM raw_conductivity_samples \
           """
try:
    df = pd.read_sql(read_sql, engine)
    print(f"Loaded {len(df)} rows from database.")
except Exception as e:
    print(f"Database connection failed: {e}")
    exit(1)

# =========================
# 2. Define prompt (translation only)
# =========================
TRANSLATION_PROMPT = """You are a materials science data normalization assistant.

Your task is to rewrite raw material descriptions into ONE concise, neutral, technical English sentence.

Strict rules:
- Do NOT add or infer any information
- Preserve chemical formulas exactly as written
- Do NOT expand abbreviations (e.g., YSZ must stay YSZ)
- If purity or concentration is not explicitly stated, do not mention it
- If supplier is not stated, do not invent one
- Do NOT mention applications, performance, or properties
- Use neutral scientific tone
- Output exactly ONE sentence without quotes.

Examples:

Input: YSZ，商品化材料(Toyo Soda)，>99%
Output: Commercial YSZ powder supplied by Toyo Soda with purity higher than 99%.

Input: Sc2O3, 商品化材料(Adventech, Korea)，6.8 mol%
Output: Commercial Sc2O3 powder supplied by Adventech (Korea) with a concentration of 6.8 mol%.

Input: {input_text}
Output:"""

prompt = PromptTemplate(
    input_variables=["input_text"],
    template=TRANSLATION_PROMPT
)

# =========================
# 3. LLM Setup
# =========================
llm = ChatOpenAI(
    api_key=API_KEY,
    model=model_name, # Recommended: use the latest model
    temperature=0.0
)

chain = prompt | llm

# =========================
# 4. Batch processing
# =========================
results = []

print("Starting translation...")
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    text_input = str(row['material_source_and_purity']).strip()
    sample_id = row['sample_id']

    # Result dict: set synthesis_method and processing_route to None (NULL)
    row_result = {
        "sample_id": sample_id,
        "material_source_and_purity": "",
        "synthesis_method": None,
        "processing_route": None
    }

    # Empty-value check
    if not text_input or text_input.lower() == 'nan':
        results.append(row_result)
        continue

    try:
        # Run translation
        response = chain.invoke({"input_text": text_input})
        translation = response.content.strip()

        row_result["material_source_and_purity"] = translation
        results.append(row_result)

    except Exception as e:
        print(f"Error processing ID {sample_id}: {e}")
        # On error, keep the original text to avoid aborting the run
        row_result["material_source_and_purity"] = text_input
        results.append(row_result)

# =========================
# 5. Write to database
# =========================
if results:
    result_df = pd.DataFrame(results)

    # Ensure column order
    cols = ["sample_id", "material_source_and_purity", "synthesis_method", "processing_route"]
    result_df = result_df[cols]

    print("Writing to database table 'tmp_translate_result'...")

    try:
        result_df.to_sql(
            name='tmp_translate_result',
            con=engine,
            if_exists='append',
            index=False,
            chunksize=1000 # If the dataset is large, writing in chunks is safer
        )
        print("Done! Check table 'tmp_translate_result'.")
    except Exception as e:
        print(f"Error writing to database: {e}")
else:
    print("No data processed.")
