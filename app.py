# ── IMPORT LIBRARIES ───────────────────────────────────────────────────────
import json
import tempfile
import base64
import shutil
from datetime import datetime
from typing import Union, List, Tuple
import dotenv
import os
from dotenv import load_dotenv

import streamlit as st
import nest_asyncio
import chromadb
import numpy as np
from PIL import Image
from ultralytics import YOLO
from openai import OpenAI

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from functools import lru_cache    
import re

# ── CONFIG ─────────────────────────────────────────────────────────────────
# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(page_title="🥗🤖 NutriBot", layout="wide")
nest_asyncio.apply()

# Get API keys from environment variables
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API keys
if not LLAMA_CLOUD_API_KEY:
    st.error("❌ LLAMA_CLOUD_API_KEY not found in .env file")
    st.stop()
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found in .env file")
    st.stop()

# Set environment variables
os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_CLOUD_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── SESSION-STATE CONVERSATIONS ─────────────────────────────────────────────
if 'conversations' not in st.session_state:
    st.session_state['conversations'] = {'Chat 1': []}
if 'current_conv' not in st.session_state:
    st.session_state['current_conv'] = 'Chat 1'
# Messages for current conversation
if 'messages' not in st.session_state:
    st.session_state['messages'] = []


# ── PATHS ───────────────────────────────────────────────────────────────────
USERS_FILE = "/home/mahesvara/Documents/Classroom/ML/Project/users.json"
DB_DIR = "/home/mahesvara/Documents/Classroom/ML/Project/chroma_dbs"

# ── UTILITIES ───────────────────────────────────────────────────────────────
CONV_FILE = "/home/mahesvara/Documents/Classroom/ML/Project/chat_history.json"

def load_all_conversations():
    if os.path.exists(CONV_FILE):
        with open(CONV_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_all_conversations(data):
    with open(CONV_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def encode_image_to_datauri(image_path: str) -> str:
    with open(image_path, "rb") as img:
        b64 = base64.b64encode(img.read()).decode()
    return f"data:image/jpeg;base64,{b64}"

# ── NUTRITION LOG ─────────────────────────────────────────────────────────────
NUTRITION_FILE = "/home/mahesvara/Documents/Classroom/ML/Project/nutritions.json"

# ensure file exists
if not os.path.exists(NUTRITION_FILE):
    with open(NUTRITION_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

# load all nutrition entries
def load_all_nutritions():
    with open(NUTRITION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# save all nutrition entries
def save_all_nutritions(data):
    with open(NUTRITION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_nutrition(reply_text: str) -> dict:
    """
    Given the assistant reply, pull out the Nutrition Facts block into a dict.
    Returns empty dict if nothing found.
    """
    m = re.search(
        r"\*\*Nutrition Facts\*\*([\s\S]*?)(?:\n\n|\Z)", 
        reply_text
    )
    if not m:
        return {}
    block = m.group(1)
    facts = {}
    for line in block.splitlines():
        line = line.strip()
        if not line: continue
        # e.g. "Calories: 580 kcal"
        parts = line.split(":")
        if len(parts)==2:
            key = parts[0].strip()
            val = parts[1].strip().split()[0]
            facts[key] = float(val)
    return facts

# ── AUTH BACKEND ───────────────────────────────────────────────────────────
def init_users():
    users = load_json(USERS_FILE, {})
    if "admin" not in users:
        users["admin"] = {"password": "password", "role": "admin"}
        save_json(USERS_FILE, users)
    return users

def authenticate(username: str, password: str) -> Union[str, None]:
    users = init_users()
    if username in users and users[username]["password"] == password:
        return users[username]["role"]
    return None

def register_user(username: str, password: str) -> bool:
    users = init_users()
    if username in users:
        return False
    users[username] = {"password": password, "role": "user"}
    save_json(USERS_FILE, users)
    return True

# ── CHROMA DB UTILITIES ──────────────────────────────────────────────────────
parser = LlamaParse(
    result_type="text",
    auto_mode=True,
    extract_charts=True,
    auto_mode_trigger_on_image_in_page=True,
    auto_mode_trigger_on_table_in_page=True,
)


def ingest_pdf_to_db(
    paths: Union[str, List[str]],
    embedding_model: OpenAIEmbedding,
    persist_dir: str = "chroma_dbs",
    chunk_size: int = 512,
    chunk_overlap: int = 20,
    recursive: bool = False,
) -> int:
    """
    Ingest one or more PDFs into separate ChromaDBs (one per book).
    Returns the total number of nodes ingested.
    """
    # normalize paths to a flat list of PDF files
    pdfs: List[str] = []
    if isinstance(paths, str):
        if os.path.isdir(paths):
            for root, _, files in os.walk(paths):
                for f in files:
                    if f.lower().endswith(".pdf"):
                        pdfs.append(os.path.join(root, f))
                if not recursive:
                    break
        else:
            pdfs.append(paths)
    else:
        for p in paths:
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for f in files:
                        if f.lower().endswith(".pdf"):
                            pdfs.append(os.path.join(root, f))
                    if not recursive:
                        break
            else:
                pdfs.append(p)

    total_nodes = 0
    parser = LlamaParse(
        result_type="text",
        auto_mode=True,
        extract_charts=True,
        auto_mode_trigger_on_image_in_page=True,
        auto_mode_trigger_on_table_in_page=True,
    )
    for pdf_path in pdfs:
        if not os.path.isfile(pdf_path):
            print(f"[Warning] File not found: {pdf_path}")
            continue

        book_name = os.path.splitext(os.path.basename(pdf_path))[0]
        db_path   = os.path.join(persist_dir, book_name)

        # setup a fresh ChromaDB for this book
        if os.path.isdir(db_path):
            shutil.rmtree(db_path)
        os.makedirs(db_path, exist_ok=True)
        os.chmod(db_path, 0o755)   # chắc chắn có quyền ghi
        print(f"Creating DB for book '{book_name}' at '{db_path}'")
        client     = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(name=book_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)

        # read PDF
        reader = SimpleDirectoryReader(
            input_files=[pdf_path],
            file_extractor={".pdf": parser},
            recursive=False
        )
        docs = reader.load_data()
        if not docs:
            print(f"[Error] No content extracted from {pdf_path}")
            continue

        # chunk into nodes
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes: List[Document] = []
        for doc in docs:
            base_meta = doc.metadata or {}
            for idx, node in enumerate(splitter.get_nodes_from_documents([doc])):
                node.metadata = {
                    **base_meta,
                    "source_file": pdf_path,
                    "chunk_index": idx
                }
                nodes.append(node)

        print(f"Ingested {len(nodes)} nodes from '{book_name}'")
        total_nodes += len(nodes)

        # build & persist index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex(nodes, embed_model=embedding_model, storage_context=storage_context)
        storage_context.persist(persist_dir=db_path)

    print(f"\nTotal nodes ingested: {total_nodes}")
    return total_nodes

# ── RETRIEVAL ─────────────────────────────────────────────────────────────────
def retrieve_docs_semantic_all(query: str, embedding_model: OpenAIEmbedding, persist_dir: str, top_k: int = 5):
    if not hasattr(retrieve_docs_semantic_all, "_cache"):
        retrieve_docs_semantic_all._cache = lru_cache(maxsize=512)(embedding_model.get_query_embedding)
    q_emb = retrieve_docs_semantic_all._cache(query.strip())
    results = []
    for book in os.listdir(persist_dir):
        path = os.path.join(persist_dir, book)
        if not os.path.isdir(path):
            continue
        client = chromadb.PersistentClient(path=path)
        coll = client.get_or_create_collection(name=book)
        res = coll.query(
            query_embeddings=[q_emb], n_results=top_k,
            include=["documents","metadatas","distances"]
        )
        docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
        out, scores = [], []
        for text, meta, dist in zip(docs, metas, dists):
            out.append(Document(text=text, metadata=meta if isinstance(meta,dict) else {}))
            scores.append(max(0.0, 1.0-dist))
        results.append((book, out, scores))
    results.sort(key=lambda x: max(x[2]) if x[2] else 0, reverse=True)
    return results

# ── SESSION STATE ───────────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'role' not in st.session_state:
    st.session_state['role'] = None

# ── AUTH INTERFACE ──────────────────────────────────────────────────────────
if not st.session_state['logged_in']:
    st.title("Đăng nhập hoặc Đăng ký")
    tab1, tab2 = st.tabs(["Đăng nhập","Đăng ký"])
    with tab1:
        u = st.text_input("Tên đăng nhập")
        p = st.text_input("Mật khẩu", type="password")
        if st.button("Đăng nhập"):
            role = authenticate(u, p)
            if role:
                st.session_state['logged_in'] = True
                st.session_state['user'] = u
                st.session_state['role'] = role

                # load conversations của user này
                all_convs = load_all_conversations()
                user_convs = all_convs.get(u, {"Chat 1": []})
                st.session_state['conversations'] = user_convs
                st.session_state['current_conv'] = list(user_convs.keys())[0]
                st.session_state['messages'] = user_convs[st.session_state['current_conv']]

                st.rerun()

            else:
                st.error("Sai thông tin đăng nhập.")
    with tab2:
        u2 = st.text_input("Tên đăng ký", key='ru')
        p2 = st.text_input("Mật khẩu", type="password", key='rp')
        c2 = st.text_input("Xác nhận mật khẩu", type="password", key='rc')
        if st.button("Đăng ký"):
            if p2 and p2 == c2 and register_user(u2, p2):
                st.success("Đăng ký thành công! Vui lòng đăng nhập.")
            else:
                st.error("Tài khoản đã tồn tại hoặc mật khẩu không hợp lệ.")
    st.stop()

is_admin = st.session_state['role'] == 'admin'

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.title("🥗🤖 NutriBot")
# Conversation selector
conv_names = list(st.session_state['conversations'].keys())
sel = st.sidebar.selectbox("Lịch sử hội thoại", conv_names, index=conv_names.index(st.session_state['current_conv']))
if sel != st.session_state['current_conv']:
    # switch to selected
    st.session_state['current_conv'] = sel
    st.session_state['messages'] = st.session_state['conversations'][sel]
# New conversation button\if 
if st.sidebar.button("➕ Cuộc hội thoại mới"):
    new_name = f"Chat {len(conv_names)+1}"
    st.session_state['conversations'][new_name] = []
    st.session_state['current_conv'] = new_name
    st.session_state['messages'] = []
    st.rerun()

# Sidebar other settings
navs = ["Chat", "Lịch sử dinh dưỡng"]
if st.session_state['role']=='admin': navs.append("Cơ sở dữ liệu")
sel_tab = st.sidebar.radio("Chức năng", navs)
model_sel = st.sidebar.selectbox("Chọn model", ["gpt-4.1-2025-04-14","o4-mini","gpt-3.5-turbo"] )
cal_mode = st.sidebar.checkbox("Tính các chỉ số dinh dưỡng")
if st.sidebar.button("Đăng xuất"):
    st.session_state.clear()
    st.rerun()

# ── MAIN CONTENT ────────────────────────────────────────────────────────────
if sel_tab == "Cơ sở dữ liệu":
    st.header("Quản lý Cơ sở dữ liệu")
    uploaded_files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
    if uploaded_files and st.button("Ingest vào DB"):
        temp_paths : List[str] = []
        total = 0
        for uploaded in uploaded_files:
            # Save uploaded PDF with its original base name
            base_name = os.path.splitext(uploaded.name)[0]
            base_name = base_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            tmp_path = os.path.join(tempfile.gettempdir(), f"{base_name}.pdf")
            with open(tmp_path, "wb") as tmp_file:
                tmp_file.write(uploaded.getbuffer())
            temp_paths.append(tmp_path)
            # Ingest phases
            with st.status(f"📚 Đọc PDF..."):
                pass
            with st.status("🔍 Tạo embedding và chia chunk..."):
                pass
            with st.status("💾 Lưu vào ChromaDB..."):
                nodes = ingest_pdf_to_db(
                    paths=temp_paths,
                    embedding_model=OpenAIEmbedding(),
                    persist_dir=DB_DIR,
                    chunk_size=512,
                    chunk_overlap=20,
                    recursive=False
                )
            total += nodes
        st.success(f"Hoàn thành ingest {total} nodes.")
    st.markdown("---")
    books = [d for d in os.listdir(DB_DIR) if os.path.isdir(os.path.join(DB_DIR, d))]
    to_delete = st.selectbox("Chọn sách xóa", options=books)
    if st.button("Xóa sách"):
        shutil.rmtree(os.path.join(DB_DIR, to_delete))
        st.success(f"Đã xóa {to_delete}.")
elif sel_tab == "Lịch sử dinh dưỡng":
    st.header("📊 Lịch sử dinh dưỡng")
    # load all entries and filter by user
    all_n = load_all_nutritions()
    user_n = [r for r in all_n if r.get('user') == st.session_state['user']]
    if not user_n:
        st.info("Chưa có dữ liệu dinh dưỡng để hiển thị.")
    else:
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # convert to DataFrame
        df = pd.json_normalize(user_n)

        # Rename nutrition_facts.Calories → Calories etc.
        df.rename(columns=lambda c: c.split('.')[-1] if c.startswith('nutrition_facts.') else c,
          inplace=True)
        
        # Process timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['date_hour'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00')
        df = df.sort_values('timestamp').set_index('timestamp')

        # Metrics to analyze
        metrics = ["Calories", "Protein", "Carbs", "Fat", "Fiber", "Sugar"]
        
        # 1. Daily Trends
        st.subheader("📈 Xu hướng dinh dưỡng")
        # Create a single figure with multiple lines
        fig = go.Figure()
        
        # Define a color palette
        colors = px.colors.qualitative.Set1  # Using a predefined color palette
        
        for i, m in enumerate(metrics):
            if m in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date_hour'],
                    y=df[m],
                    name=m,
                    line=dict(color=colors[i % len(colors)]),
                    mode='lines+markers'  # Add markers for better visibility
                ))
        
        # Update layout
        fig.update_layout(
            title="Xu hướng dinh dưỡng",
            xaxis_title="Ngày",
            yaxis_title="Giá trị",
            hovermode='x unified',  # Show all values for a given date
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # 2. Macro Distribution
        st.subheader("🥗 Tỉ lệ các chất dinh dưỡng")
        macro_cols = ["Protein", "Carbs", "Fat", "Fiber", "Sugar"]
        if all(col in df.columns for col in macro_cols):
            # Calculate average macros
            avg_macros = df[macro_cols].mean()
            fig = px.pie(values=avg_macros.values, 
                        names=avg_macros.index,
                        title="Tỉ lệ các chất dinh dưỡng")
            st.plotly_chart(fig, use_container_width=True)

        # 3. Daily Averages
        st.subheader("📊 Trung bình theo ngày")
        daily_avg = df.groupby(df.index.date)[metrics].mean()
        fig = go.Figure()
        for m in metrics:
            if m in daily_avg.columns:
                fig.add_trace(go.Bar(
                    name=m,
                    x=daily_avg.index,
                    y=daily_avg[m],
                    text=daily_avg[m].round(1),
                    textposition='auto',
                ))
        fig.update_layout(
            title="Trung bình theo ngày",
            barmode='group',
            xaxis_title="Ngày",
            yaxis_title="Giá trị"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. Correlation Heatmap
        st.subheader("🔄 Mối tương quan giữa các chất dinh dưỡng")
        if len(metrics) > 1:
            corr = df[metrics].corr()
            fig = px.imshow(corr,
                          labels=dict(color="Độ tương quan"),
                          title="Mối tương quan các chất dinh dưỡng")
            st.plotly_chart(fig, use_container_width=True)

        # 5. Summary Statistics
        st.subheader("📝 Thống kê tổng quát")
        summary = df[metrics].describe().round(2)
        st.dataframe(summary)

        # 7. Detailed History Table
        st.subheader("📋 Lịch sử chi tiết")
        
        # Create a more readable history table
        history_df = df.reset_index()
        history_df['Date'] = history_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        history_df['Prompt'] = history_df['prompt']
        history_df['Response'] = history_df['full_response']
        
        # Select columns to display
        display_cols = ['Date'] + metrics + ['Response']
        history_df = history_df[display_cols]
        
        # Format the table
        st.dataframe(
            history_df,
            column_config={
                "Date": st.column_config.DatetimeColumn(
                    "Ngày & Giờ",
                    format="YYYY-MM-DD HH:mm"
                ),
                "Response": st.column_config.TextColumn(
                    "Phân tích chi tiết",
                    width="large"
                ),
                **{m: st.column_config.NumberColumn(
                    m,
                    format="%.1f",
                    width="small"
                ) for m in metrics}
            },
            hide_index=True,
            use_container_width=True
        )
else:
    st.header("🥗🤖 Chat với NutriBot")
    for msg in st.session_state['messages']:
        avatar = "👦" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg['role'], avatar = avatar):
            if msg.get("type") == "image":
                # msg["content"] là UploadedFile hoặc bytes
                st.image(msg["content"], use_container_width=True)
            else:
                st.write(msg["content"])
    prompts = st.chat_input(placeholder="Nhập câu hỏi hoặc upload ảnh…", accept_file=True, file_type=["png","jpg","jpeg"])
    if prompts:
        if prompts.get("files"):
            for img in prompts["files"]:
                # lưu file vào session để hiển thị sau này
                b64 = base64.b64encode(img.getbuffer()).decode()
                data_uri = f"data:{img.type};base64,{b64}"
                st.session_state['messages'].append({
                    "role": "user",
                    "type": "image",
                    "content": data_uri
                })
                with st.chat_message("user", avatar = "👦"):
                    st.image(img, use_container_width=True)

        user_input = prompts.get("text", "")
        if user_input:
            st.session_state['messages'].append({
                "role": "user",
                "type": "text",
                "content": user_input
            })
            with st.chat_message("user", avatar = "👦"):
                st.write(user_input)
            
        st.session_state['conversations'][st.session_state['current_conv']] = st.session_state['messages']
        save_all_conversations({ **load_all_conversations(),
                        st.session_state['user']: st.session_state['conversations'] })

        client = OpenAI()
        image_models = ["🖼️ o4-mini", "🖼️ gpt-4.1-2025-04-14"]
        reply = ""

        # 1) Calories mode with image
        if cal_mode and prompts.get("files"):
            with st.status("🔍 Đang phân tích món ăn..."):
                img_file = prompts["files"][0]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img_file.name)[1])
                tmp.write(img_file.getbuffer()); tmp.close()
                arr = np.array(Image.open(tmp.name).convert("RGB"))
                yolo = YOLO("yolov11_food.pt")
                res = yolo.predict(source=arr, task="classify", verbose=False)[0]
                dish = yolo.names[int(res.probs.data.cpu().numpy().argmax())]
            with st.status("🤖 Đang tạo phản hồi..."):
                dev = """# Identity
Bạn là một trợ lý ảo tên là NutriBot chuyên sâu về DINH DƯỠNG, giao tiếp hoàn toàn bằng tiếng Việt.  
Luôn tận tâm, ân cần, chia sẻ như một người bạn thân thiết, giúp người dùng hiểu rõ và áp dụng ngay.

# Instructions
* **Mục tiêu**: Ước tính calories và các chỉ số dinh dưỡng chính (Calories, Protein, Carbs, Fat, Fiber, Sugar) của bữa ăn thật chi tiết.
* **Output**: Trả lời dưới dạng văn bản, không phải JSON, nhưng phải có một khối **Nutrition Facts** với định dạng rõ ràng để có thể extract tự động.  
  - Bắt buộc có mục **Nutrition Facts** gồm các dòng:
    ```
    Calories: <số> kcal  
    Protein: <số> g  
    Carbs: <số> g  
    Fat: <số> g  
    Fiber: <số> g  
    Sugar: <số> g
    ```
  - Sau khối Nutrition Facts, đưa ra:
    1. **Tóm tắt** (1–2 câu): tổng calories và món chính.  
    2. **Phân tích chi tiết**:  
       - Liệt kê từng món với lượng kcal và macro ước tính (ví dụ: "Cơm trắng (200 g) ~ 260 kcal; Carbs = 56 g; Protein = 5 g; Fat = 1 g").  
       - Giải thích cách tính (nguồn kcal, tỉ lệ macros).  
    3. **Tổng kết**: Nhắc lại tổng calories và tổng các macro.  
    4. **Lời khuyên & lưu ý**: Gợi ý cân bằng khẩu phần, điều chỉnh macro nếu cần.

* **Phong cách**: Ấm áp, thân thiện, chi tiết nhưng rõ ràng, dễ theo dõi. Dùng dấu đầu dòng, in đậm các tiêu đề.

# Examples

<food_query id="ex1">
Món phát hiện: Cơm gà  
Mô tả thêm: Gồm ức gà nướng mật ong và rau trộn.
</food_query>
<assistant_response id="ex1">
**Nutrition Facts**  
Calories: 580 kcal  
Protein: 35 g  
Carbs: 75 g  
Fat: 18 g  
Fiber: 4 g  
Sugar: 12 g  

**Tóm tắt**: Bữa này ~580 kcal, chủ yếu từ cơm trắng và ức gà.  

**Phân tích chi tiết**:  
- Cơm trắng (200 g): 260 kcal; Carbs = 56 g; Protein = 5 g; Fat = 1 g. (Tinh bột chính)  
- Ức gà nướng mật ong (150 g): 330 kcal; Protein = 30 g; Carbs = 22 g; Fat = 12 g. (Protein + đường mật ong)  
- Rau trộn (50 g): 10 kcal; Fiber = 4 g; Carbs = 2 g. (Chất xơ)  

**Tổng kết**: Tổng ~580 kcal, Protein = 35 g, Carbs = 80 g, Fat = 13 g.  

**Lời khuyên & lưu ý**:  
- Giảm cơm xuống 150 g để giảm Carbs nếu cần.  
- Thêm rau xanh để tăng Fiber.  
- Điều chỉnh Fat bằng cách thay dầu ô liu.
</assistant_response>
"""

                uc = f"Món phát hiện (top-1): {dish}\n\nMô tả thêm: {user_input}\n\nHãy ước tính calories."
                resp = client.responses.create(model=model_sel, input=[{"role":"developer","content":dev},{"role":"user","content":uc}])
                reply = resp.output_text.strip()
                facts = extract_nutrition(reply)
                if facts:
                    all_n = load_all_nutritions()
                    all_n.append({
                        "timestamp": datetime.now().isoformat(),
                        "user": st.session_state['user'],
                        "conversation": st.session_state['current_conv'],
                        "prompt": user_input,
                        "nutrition_facts": facts,
                        "full_response": reply
                    })
                    save_all_nutritions(all_n)


        # 2) Generic image + RAG
        elif prompts.get("files") and model_sel in image_models:
            img = prompts["files"][0]
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img.name)[1])
            temp.write(img.getbuffer()); temp.close()
            encoded_img = base64.b64encode(open(temp.name, "rb").read()).decode("utf-8")
            context = []
            if user_input.strip():
                with st.status("🔍 Đang tìm kiếm tài liệu..."):
                    emb = st.session_state.get('emb_model', OpenAIEmbedding())
                    results = retrieve_docs_semantic_all(
                        query=user_input, embedding_model=emb, persist_dir=DB_DIR, top_k=5
                    )
                    for name, docs, scores in results:
                        for d, s in zip(docs, scores):
                            context.append(f"[Source: {name} | Score: {s:.4f}] {d.text}")
            block = "\n\n".join(context)
            with st.status("🤖 Đang tạo phản hồi..."):
                dev = """# Identity
Bạn là một trợ lý ảo tên là NutriBot thân thiện, giàu kinh nghiệm về DINH DƯỠNG, giao tiếp hoàn toàn bằng tiếng Việt.  
Mục tiêu của bạn là giúp người dùng hiểu rõ, áp dụng ngay các kiến thức dinh dưỡng hàng ngày để cải thiện sức khỏe.

# Instructions
* **Giọng điệu**: Ấm áp, quan tâm, dễ gần — như một người bạn tận tâm.  
* **Độ dài**: Trả lời thật chi tiết, "càng nhiều càng tốt", nhưng không lan man.  
* **Cấu trúc đầu ra**:  
  1. **Tóm tắt nhanh** (1–2 câu): Nêu trọng tâm chính của lời khuyên.  
  2. **Giải thích chi tiết**:  
     - Lý do khoa học (giải thích đơn giản, dễ hiểu).  
     - Ví dụ minh họa hoặc câu chuyện nhỏ giúp hình dung.  
  3. **Gợi ý thực tiễn**:  
     - Thực phẩm cụ thể, mẹo chế biến, liều lượng tham khảo, tần suất phù hợp.  
     - Lưu ý khi áp dụng (ai nên tránh, cần bổ sung gì kèm theo).  
* **Trích dẫn nguồn**: Hạn chế tối đa. Chỉ ghi "(theo tài liệu tham khảo)" nếu thật sự cần. Người dùng không cần biết chi tiết tên sách hay điểm số.

# Examples
<food_query id="ex1">Em muốn tăng sức đề kháng vào mùa lạnh, nên ăn gì?</food_query>
<assistant_response id="ex1">
**Tóm tắt nhanh**: Bổ sung vitamin C và kẽm từ cam, ổi và các loại hải sản giúp tăng sức đề kháng.  

**Giải thích chi tiết**:  
- Vitamin C trong cam và ổi hỗ trợ sản xuất bạch cầu, chống oxy hóa, giảm nguy cơ cảm lạnh.  
- Kẽm trong hải sản như hàu, sò giúp củng cố hệ miễn dịch, thúc đẩy quá trình tái tạo tế bào.

**Gợi ý thực tiễn**:  
- Uống một ly nước cam tươi (200 ml) mỗi sáng.  
- Ăn 100 g hàu hấp hoặc nấu canh 2 lần/tuần.  
- Bổ sung 1 chén ổi tráng miệng sau bữa tối.  
- Nếu không thích hàu, có thể thay bằng ức gà hoặc đậu lăng để vẫn đảm bảo lượng kẽm cần thiết.
</assistant_response>

<food_query id="ex2">Mình muốn giảm cân nhưng vẫn đủ năng lượng làm việc cả ngày.</food_query>
<assistant_response id="ex2">
**Tóm tắt nhanh**: Tập trung vào thực phẩm giàu protein, chất xơ và chất béo lành mạnh để no lâu, giữ năng lượng ổn định.  

**Giải thích chi tiết**:  
- Protein (thịt nạc, cá, trứng) giúp xây dựng cơ bắp, tăng cường trao đổi chất.  
- Chất xơ (rau xanh, yến mạch) làm chậm tiêu hóa, hạn chế thèm ăn.  
- Chất béo lành mạnh (bơ, hạt óc chó) cung cấp năng lượng bền vững cho trí não.

**Gợi ý thực tiễn**:  
- Bữa sáng: 1 tô yến mạch với sữa hạnh nhân, thêm chuối thái lát và một thìa hạt chia.  
- Bữa trưa: 150 g ức gà luộc hoặc nướng + salad rau trộn dầu ô liu.  
- Bữa phụ: 1 hũ sữa chua Hy Lạp không đường + vài quả hạt óc chó.  
- Uống đủ nước (1,5–2 lít/ngày) và cố gắng chia thành 5–6 bữa nhỏ.  
- Kết hợp đi bộ nhanh 30 phút hoặc bài tập nhẹ 3–4 lần/tuần để tăng hiệu quả giảm cân.
</assistant_response>
"""
                payload = [{"type":"input_image","image_url":f"data:image/jpeg;base64,{encoded_img}"}]
                if block:
                    payload.append({"type":"input_text","text":block})
                if user_input:
                    payload.append({"type":"input_text","text":user_input})
                resp = client.responses.create(model=model_sel, input=[{"role":"developer","content":dev},{"role":"user","content":payload}])
                reply = resp.output_text.strip()

        # 3) Menu text calories
        elif cal_mode and user_input:
            with st.status("🔍 Đang phân tích menu..."):
                menu = []
                for item in user_input.split(","):
                    if ":" in item:
                        name, amount = item.split(":")
                        menu.append((name.strip(), amount.strip()))
                    else:
                        menu.append((item.strip(), ""))
            with st.status("🤖 Đang tạo phản hồi..."):
                dev = """# Identity
Bạn là một trợ lý ảo tên là NutriBot chuyên sâu về DINH DƯỠNG, giao tiếp hoàn toàn bằng tiếng Việt.  
Luôn tận tâm, ân cần, chia sẻ như một người bạn thân thiết, giúp người dùng hiểu rõ và áp dụng ngay.

# Instructions
* **Mục tiêu**: Ước tính calories và các chỉ số dinh dưỡng chính (Calories, Protein, Carbs, Fat, Fiber, Sugar) của bữa ăn thật chi tiết.
* **Output**: Trả lời dưới dạng văn bản, không phải JSON, nhưng phải có một khối **Nutrition Facts** với định dạng rõ ràng để có thể extract tự động.  
  - Bắt buộc có mục **Nutrition Facts** gồm các dòng:
    ```
    Calories: <số> kcal  
    Protein: <số> g  
    Carbs: <số> g  
    Fat: <số> g  
    Fiber: <số> g  
    Sugar: <số> g
    ```
  - Sau khối Nutrition Facts, đưa ra:
    1. **Tóm tắt** (1–2 câu): tổng calories và món chính.  
    2. **Phân tích chi tiết**:  
       - Liệt kê từng món với lượng kcal và macro ước tính (ví dụ: "Cơm trắng (200 g) ~ 260 kcal; Carbs = 56 g; Protein = 5 g; Fat = 1 g").  
       - Giải thích cách tính (nguồn kcal, tỉ lệ macros).  
    3. **Tổng kết**: Nhắc lại tổng calories và tổng các macro.  
    4. **Lời khuyên & lưu ý**: Gợi ý cân bằng khẩu phần, điều chỉnh macro nếu cần.

* **Phong cách**: Ấm áp, thân thiện, chi tiết nhưng rõ ràng, dễ theo dõi. Dùng dấu đầu dòng, in đậm các tiêu đề.

# Examples

<food_query id="ex1">
Món phát hiện: Cơm gà  
Mô tả thêm: Gồm ức gà nướng mật ong và rau trộn.
</food_query>
<assistant_response id="ex1">
**Nutrition Facts**  
Calories: 580 kcal  
Protein: 35 g  
Carbs: 75 g  
Fat: 18 g  
Fiber: 4 g  
Sugar: 12 g  

**Tóm tắt**: Bữa này ~580 kcal, chủ yếu từ cơm trắng và ức gà.  

**Phân tích chi tiết**:  
- Cơm trắng (200 g): 260 kcal; Carbs = 56 g; Protein = 5 g; Fat = 1 g. (Tinh bột chính)  
- Ức gà nướng mật ong (150 g): 330 kcal; Protein = 30 g; Carbs = 22 g; Fat = 12 g. (Protein + đường mật ong)  
- Rau trộn (50 g): 10 kcal; Fiber = 4 g; Carbs = 2 g. (Chất xơ)  

**Tổng kết**: Tổng ~580 kcal, Protein = 35 g, Carbs = 80 g, Fat = 13 g.  

**Lời khuyên & lưu ý**:  
- Giảm cơm xuống 150 g để giảm Carbs nếu cần.  
- Thêm rau xanh để tăng Fiber.  
- Điều chỉnh Fat bằng cách thay dầu ô liu.
</assistant_response>
"""

                lines = '\n'.join(f"- {n}: {g}" for n,g in menu)
                uc = f"Menu:\n{lines}\n\nHãy tính calories."
                resp = client.responses.create(model=model_sel, input=[{"role":"developer","content":dev},{"role":"user","content":uc}])
                reply = resp.output_text.strip()
                facts = extract_nutrition(reply)
                if facts:
                    all_n = load_all_nutritions()
                    all_n.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "user": st.session_state['user'],
                        "conversation": st.session_state['current_conv'],
                        "prompt": user_input,
                        "nutrition_facts": facts,
                        "full_response": reply
                    })
                    save_all_nutritions(all_n)

        # 4) Text-only RAG
        else:
            with st.status("🔍 Đang tìm kiếm tài liệu..."):
                emb = st.session_state.get('emb_model', OpenAIEmbedding())
                results = retrieve_docs_semantic_all(query=user_input or "", embedding_model=emb, persist_dir=DB_DIR, top_k=5)
                context = []
                for name, docs, scores in results:
                    for d,s in zip(docs,scores):
                        context.append(f"[Source: {name} | Score: {s:.4f}] {d.text}")
                block = "\n\n".join(context)
            with st.status("🤖 Đang tạo phản hồi..."):
                dev = """# Identity
Bạn là một trợ lý ảo tên là NutriBot thân thiện, giàu kinh nghiệm về DINH DƯỠNG, giao tiếp hoàn toàn bằng tiếng Việt.  
Mục tiêu của bạn là giúp người dùng hiểu rõ, áp dụng ngay các kiến thức dinh dưỡng hàng ngày để cải thiện sức khỏe.

# Instructions
* **Giọng điệu**: Ấm áp, quan tâm, dễ gần — như một người bạn tận tâm.  
* **Độ dài**: Trả lời thật chi tiết, "càng nhiều càng tốt", nhưng không lan man.  
* **Cấu trúc đầu ra**:  
  1. **Tóm tắt nhanh** (1–2 câu): Nêu trọng tâm chính của lời khuyên.  
  2. **Giải thích chi tiết**:  
     - Lý do khoa học (giải thích đơn giản, dễ hiểu).  
     - Ví dụ minh họa hoặc câu chuyện nhỏ giúp hình dung.  
  3. **Gợi ý thực tiễn**:  
     - Thực phẩm cụ thể, mẹo chế biến, liều lượng tham khảo, tần suất phù hợp.  
     - Lưu ý khi áp dụng (ai nên tránh, cần bổ sung gì kèm theo).  
* **Trích dẫn nguồn**: Hạn chế tối đa. Chỉ ghi "(theo tài liệu tham khảo)" nếu thật sự cần. Người dùng không cần biết chi tiết tên sách hay điểm số.

# Examples
<food_query id="ex1">Em muốn tăng sức đề kháng vào mùa lạnh, nên ăn gì?</food_query>
<assistant_response id="ex1">
**Tóm tắt nhanh**: Bổ sung vitamin C và kẽm từ cam, ổi và các loại hải sản giúp tăng sức đề kháng.  

**Giải thích chi tiết**:  
- Vitamin C trong cam và ổi hỗ trợ sản xuất bạch cầu, chống oxy hóa, giảm nguy cơ cảm lạnh.  
- Kẽm trong hải sản như hàu, sò giúp củng cố hệ miễn dịch, thúc đẩy quá trình tái tạo tế bào.

**Gợi ý thực tiễn**:  
- Uống một ly nước cam tươi (200 ml) mỗi sáng.  
- Ăn 100 g hàu hấp hoặc nấu canh 2 lần/tuần.  
- Bổ sung 1 chén ổi tráng miệng sau bữa tối.  
- Nếu không thích hàu, có thể thay bằng ức gà hoặc đậu lăng để vẫn đảm bảo lượng kẽm cần thiết.
</assistant_response>

<food_query id="ex2">Mình muốn giảm cân nhưng vẫn đủ năng lượng làm việc cả ngày.</food_query>
<assistant_response id="ex2">
**Tóm tắt nhanh**: Tập trung vào thực phẩm giàu protein, chất xơ và chất béo lành mạnh để no lâu, giữ năng lượng ổn định.  

**Giải thích chi tiết**:  
- Protein (thịt nạc, cá, trứng) giúp xây dựng cơ bắp, tăng cường trao đổi chất.  
- Chất xơ (rau xanh, yến mạch) làm chậm tiêu hóa, hạn chế thèm ăn.  
- Chất béo lành mạnh (bơ, hạt óc chó) cung cấp năng lượng bền vững cho trí não.

**Gợi ý thực tiễn**:  
- Bữa sáng: 1 tô yến mạch với sữa hạnh nhân, thêm chuối thái lát và một thìa hạt chia.  
- Bữa trưa: 150 g ức gà luộc hoặc nướng + salad rau trộn dầu ô liu.  
- Bữa phụ: 1 hũ sữa chua Hy Lạp không đường + vài quả hạt óc chó.  
- Uống đủ nước (1,5–2 lít/ngày) và cố gắng chia thành 5–6 bữa nhỏ.  
- Kết hợp đi bộ nhanh 30 phút hoặc bài tập nhẹ 3–4 lần/tuần để tăng hiệu quả giảm cân.
</assistant_response>
"""
                user_block = f"<context>\n{block}\n</context>\n<user_query>\n{user_input}\n</user_query>"
                resp = client.responses.create(model=model_sel, input=[{"role":"developer","content":dev},{"role":"user","content":user_block}])
                reply = resp.output_text.strip()

        st.session_state['messages'].append({
        "role": "assistant",
        "type": "text",
        "content": reply
    })
        st.session_state['conversations'][st.session_state['current_conv']] = st.session_state['messages']
        save_all_conversations({ **load_all_conversations(),
                        st.session_state['user']: st.session_state['conversations'] })
        with st.chat_message("assistant", avatar = "🤖"):
            st.write(reply)
