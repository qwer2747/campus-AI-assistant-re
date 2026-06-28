import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 
# ======================================

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import time

print("="*50)
print("构建Chroma知识库")
print("="*50)

# ==================== 1. 检查CSV文件 ====================
csv_file = "campus_qa.csv"
if not os.path.exists(csv_file):
    print(f"错误：找不到 {csv_file}")
    print("请确保CSV文件在当前文件夹，并且名为 campus_qa.csv")
    exit()

print(f"找到CSV文件：{csv_file}")

# ==================== 2. 加载嵌入模型 ====================
print("\n正在加载嵌入模型（首次运行会下载，约500MB）...")
start_time = time.time()

try:
    model = SentenceTransformer('BAAI/bge-small-zh')
    print(f"模型加载成功！向量维度：{model.get_sentence_embedding_dimension()}")
    print(f"⏱耗时：{time.time()-start_time:.2f}秒")
except Exception as e:
    print(f"模型加载失败：{e}")
    exit()

# ==================== 3. 读取CSV数据 ====================
print(f"\n正在读取CSV文件...")
try:
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    print(f"读取成功！共 {len(df)} 条问答")
    print("\n数据预览（前3条）：")
    for i, row in df.head(3).iterrows():
        print(f"  {i+1}. 问题：{row['question'][:30]}...")
        print(f"     答案：{row['answer'][:30]}...")
except Exception as e:
    print(f"读取CSV失败：{e}")
    print("尝试用其他编码...")
    try:
        df = pd.read_csv(csv_file, encoding='gbk')
        print("用gbk编码读取成功！")
    except:
        exit()

# ==================== 4. 初始化Chroma ====================
print(f"\n初始化Chroma客户端...")

# 创建持久化chroma客户端（数据会保存到./chroma_db文件夹）
client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "campus_qa"
try:
    client.delete_collection(collection_name)
    print(f"已删除旧集合：{collection_name}")
except:
    pass

# 创建新集合
collection = client.create_collection(
    name=collection_name,
    metadata={"description": "校园新生指南问答库", "total_qa": len(df)}
)
print(f"已创建新集合：{collection_name}")

# ==================== 5. 准备数据并向量化 ====================
print(f"\n正在准备数据并生成向量...")

# 准备文档和ID
documents = []
ids = []
for idx, row in df.iterrows():
    content = f"问题：{row['question']}\n答案：{row['answer']}"
    documents.append(content)
    ids.append(f"qa_{idx:04d}")  # 生成如 qa_0001, qa_0002 的ID

print(f"共 {len(documents)} 条文档待处理")

# 生成向量
print(f"正在生成向量（可能需要1-2分钟）...")
start_time = time.time()
embeddings = model.encode(documents, show_progress_bar=True).tolist()
print(f"向量生成完成，维度：{len(embeddings[0])}")
print(f"耗时：{time.time()-start_time:.2f}秒")

# ==================== 6. 添加到Chroma ====================
print(f"\n正在添加数据到Chroma...")

batch_size = 50
for i in range(0, len(documents), batch_size):
    end_idx = min(i + batch_size, len(documents))
    print(f"添加批次 {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({i}-{end_idx})")
    
    collection.add(
        documents=documents[i:end_idx],
        embeddings=embeddings[i:end_idx],
        ids=ids[i:end_idx]
    )

print(f"成功添加 {len(documents)} 条数据到Chroma")

# ==================== 7. 验证检索效果 ====================
print(f"\n 测试检索效果...")

test_questions = [
    "图书馆几点开门？",
    "学生卡丢了怎么办？",
    "学校有几个食堂？"
]

for test_q in test_questions:
    print(f"\n问题：{test_q}")
    
    # 生成问题向量
    test_embedding = model.encode(test_q).tolist()
    
    # 在Chroma中搜索
    results = collection.query(
        query_embeddings=[test_embedding],
        n_results=2
    )
    
    print(f"最相关的2条结果：")
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        # 显示前80个字符
        preview = doc[:80] + "..." if len(doc) > 80 else doc
        preview = preview.replace('\n', ' ')  # 把换行符换成空格
        print(f"   {i+1}. [相似度：{1-distance:.4f}] {preview}")

# ==================== 8. 完成 ====================
print(f"知识库已保存到：{os.path.abspath('./chroma_db')}")
print(f"总记录数：{collection.count()} 条")
