# 🥗🤖 NutriBot - Trợ lý dinh dưỡng thông minh

NutriBot là một ứng dụng trợ lý dinh dưỡng thông minh được xây dựng bằng Python và Streamlit. Ứng dụng giúp người dùng phân tích dinh dưỡng từ hình ảnh món ăn, tính toán calories, và cung cấp lời khuyên dinh dưỡng dựa trên cơ sở dữ liệu kiến thức.

## 🌟 Tính năng chính

- 📸 Phân tích dinh dưỡng từ hình ảnh món ăn
- 🔢 Tính toán calories và các chỉ số dinh dưỡng
- 📊 Theo dõi lịch sử dinh dưỡng với biểu đồ trực quan
- 📚 Tích hợp cơ sở dữ liệu kiến thức dinh dưỡng
- 👥 Hệ thống đăng nhập và quản lý người dùng
- 💬 Chat tương tác với trợ lý AI

## 🛠️ Yêu cầu hệ thống

- Python 3.8+
- Conda hoặc Miniconda
- CUDA (khuyến nghị cho xử lý hình ảnh)

## 📦 Cài đặt

1. **Clone repository**
```bash
git clone <repository-url>
cd Project
```

2. **Tạo môi trường Conda**
```bash
conda env create -f environment.yaml
conda activate nutribot
```

3. **Cấu trúc thư mục**
Dự án đã bao gồm sẵn cơ sở dữ liệu đã được ingest. Đảm bảo các thư mục và file sau tồn tại:
```
Project/
├── chroma_dbs/           # Thư mục chứa cơ sở dữ liệu vector đã được ingest
└── data/                 # Thư mục chứa dữ liệu người dùng
    ├── users.json        # Thông tin người dùng
    ├── chat_history.json # Lịch sử chat
    └── nutritions.json   # Dữ liệu dinh dưỡng
```

4. **Tạo các file JSON**
Tạo thư mục `data` và các file JSON với cấu trúc ban đầu:

```bash
mkdir -p data
```

a) **users.json** - File chứa thông tin người dùng:
```json
{
  "admin": {
    "password": "password",
    "role": "admin"
  }
}
```

b) **chat_history.json** - File chứa lịch sử chat:
```json
{}
```

c) **nutritions.json** - File chứa dữ liệu dinh dưỡng:
```json
[]
```

5. **Tạo file .env**
Tạo file `.env` trong thư mục `Project` với nội dung:
```
# API Keys
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DB_DIR=chroma_dbs

# Model Configuration
DEFAULT_MODEL=gpt-4.1-2025-04-14
```

6. **Cập nhật đường dẫn**
Trong file `app.py`, cập nhật các đường dẫn sau:
```python
USERS_FILE = "path/to/users.json"
DB_DIR = "path/to/chroma_dbs"
CONV_FILE = "path/to/chat_history.json"
NUTRITION_FILE = "path/to/nutritions.json"
```

## 🚀 Chạy ứng dụng

1. **Kích hoạt môi trường**
```bash
conda activate nutribot
```

2. **Chạy Streamlit**
```bash
streamlit run app.py
```

3. **Truy cập ứng dụng**
Mở trình duyệt và truy cập: `http://localhost:8501`

## 📝 Cấu trúc dự án

```
Project/
├── app.py                 # File chính của ứng dụng
├── environment.yaml       # Cấu hình môi trường Conda
├── .env                  # File chứa API keys
├── README.md             # Tài liệu hướng dẫn
├── chroma_dbs/           # Thư mục chứa cơ sở dữ liệu vector đã được ingest
└── data/                 # Thư mục chứa dữ liệu người dùng
    ├── users.json        # Thông tin người dùng
    ├── chat_history.json # Lịch sử chat
    └── nutritions.json   # Dữ liệu dinh dưỡng
```

## 🔑 Tài khoản mặc định

- **Admin**: 
  - Username: admin
  - Password: password

## ⚠️ Lưu ý quan trọng

1. Đảm bảo đã cài đặt đầy đủ các thư viện từ `environment.yaml`
2. Kiểm tra và cập nhật các API keys trong file `.env`
3. Đảm bảo các đường dẫn trong `app.py` được cấu hình đúng
4. **KHÔNG** xóa hoặc tạo mới thư mục `chroma_dbs` vì nó đã chứa dữ liệu đã được ingest
5. Cấp quyền đọc cho thư mục `chroma_dbs` và quyền ghi cho thư mục `data`
6. Đảm bảo các file JSON trong thư mục `data` có cấu trúc đúng định dạng:
   - `users.json`: phải có ít nhất tài khoản admin
   - `chat_history.json`: phải là một object JSON rỗng `{}`
   - `nutritions.json`: phải là một array JSON rỗng `[]`

## 🆘 Xử lý lỗi thường gặp

1. **Lỗi API Key**
   - Kiểm tra file `.env` đã được tạo đúng
   - Xác nhận API keys hợp lệ

2. **Lỗi đường dẫn**
   - Kiểm tra các đường dẫn trong `app.py`
   - Đảm bảo thư mục `chroma_dbs` tồn tại và có quyền đọc
   - Đảm bảo thư mục `data` tồn tại và có quyền ghi

3. **Lỗi thư viện**
   - Chạy `conda env update -f environment.yaml`
   - Kiểm tra phiên bản Python (3.8+)

4. **Lỗi cơ sở dữ liệu**
   - KHÔNG xóa hoặc tạo mới thư mục `chroma_dbs`
   - Kiểm tra quyền truy cập thư mục `chroma_dbs`
   - Đảm bảo đường dẫn `DB_DIR` trong `.env` trỏ đến đúng thư mục

5. **Lỗi JSON**
   - Kiểm tra cấu trúc các file JSON trong thư mục `data`
   - Đảm bảo `users.json` có tài khoản admin
   - Đảm bảo `chat_history.json` là object rỗng `{}`
   - Đảm bảo `nutritions.json` là array rỗng `[]`
   - Sử dụng công cụ như [JSONLint](https://jsonlint.com/) để kiểm tra cú pháp JSON

## 📄 Giấy phép

Dự án này được phát hành dưới giấy phép MIT.

## 👥 Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request để đóng góp. 