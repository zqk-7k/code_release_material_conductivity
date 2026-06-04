import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Text
import os
from dotenv import load_dotenv


load_dotenv()
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host =  os.getenv("DB_HOST", "127.0.0.1")
port = int(os.getenv("DB_PORT", "3306"))
database = os.getenv("DB_NAME")




def load_raw_data_to_mysql():
    current_dir = os.path.dirname(os.path.abspath("__file__"))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, "data")
    file_name = "sample_data.xlsx"
    data_file = os.path.join(data_dir, file_name)

    # --- Key change: dtype=str ---
    # Force all columns to be read as strings.
    # keep_default_na=False means empty Excel cells are read as empty strings "",
    # not NaN. This keeps everything as plain text, so you don't need to handle NULLs.
    df = pd.read_excel(data_file, dtype=str, keep_default_na=False)

    # 2. Column name mapping
    rename_map = {
        "序号": "sample_id",
        "文献来源": "reference",
        "原材料来源及纯度": "material_source_and_purity",
        "材料制备方法": "synthesis_method",
        "制备工艺": "processing_route",
        "热处理（烧结）温度/℃": "sintering_temperature",
        "热处理时间": "sintering_duration",
        "掺杂元素": "dopant_element",
        "掺杂元素离子半径（pm）": "dopant_ionic_radius",
        "掺杂元素价态": "dopant_valence",
        "掺杂比例\n（对应形成的氧化物占总氧化物的摩尔比）": "dopant_molar_fraction",
        "晶型(c/t/m/o)": "crystal_phase",
        "工作温度(℃)": "operating_temperature",
        "电导率(S/cm)": "conductivity",
    }

    df = df.rename(columns=rename_map)

    # 3. Select columns
    columns_in_table = [
        "sample_id", "reference", "material_source_and_purity", "synthesis_method",
        "processing_route", "sintering_temperature", "sintering_duration",
        "dopant_element", "dopant_ionic_radius", "dopant_valence",
        "dopant_molar_fraction", "crystal_phase", "operating_temperature", "conductivity",
    ]
    df = df[columns_in_table]

    # 4. Create connection
    engine = create_engine(
        f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
    )

    # 5. Write to database
    # The dtype argument tells the DB to create all columns as Text (long text),
    # even if they look numeric, and store them as text.
    # Note: if the table already exists and has int/float columns, appending may fail;
    # it's recommended to drop and recreate the table.
    df.to_sql(
        "raw_conductivity_samples",
        con=engine,
        if_exists="replace",  # Recommended: use replace to recreate the table so columns are Text
        index=False,
        dtype={col: Text for col in df.columns}  # Force all columns to be created as TEXT in MySQL
    )

    print("All data has been inserted into the database as plain text.")


if __name__ == '__main__':
    load_raw_data_to_mysql()
