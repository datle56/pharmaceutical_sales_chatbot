import numpy as np
from utils.embedding import get_embeddings

def cosine_similarity(vector1, vector2):
    # Compute cosine similarity between two vectors
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2)

def find_similar_products_manual(user_query, products, top_k=5):
    embeddings = get_embeddings()
    # Tạo vector cho truy vấn người dùng
    query_vector = embeddings.embed_query(user_query)

    # Tính toán độ tương đồng
    similarities = []
    for product in products:
        product_vector = product["vector"]
        similarity = cosine_similarity(query_vector, product_vector)

        # Sao chép lại product để tránh làm mất dữ liệu gốc
        product_copy = dict(product)
        # Bỏ vector trong bản sao (nếu bạn muốn trả về product mà không kèm vector)
        product_copy.pop('vector', None)

        similarities.append((product_copy, similarity))

    # Sắp xếp theo độ tương đồng giảm dần
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Lấy top_k
    top_products = [item[0] for item in similarities[:top_k]]

    # Trả về list _id và danh sách product
    return [product["_id"] for product in top_products], top_products

