import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage import io

# Đọc ảnh từ URL hoặc đường dẫn tệp
image_url = 'https://th.bing.com/th/id/OIP.DWB33shUjG7HFHWuksU4CgAAAA?rs=1&pid=ImgDetMain'  # Thay thế bằng URL hoặc đường dẫn ảnh của bạn
image = io.imread(image_url)

# Chuyển ảnh về định dạng vector (flatten) với mỗi pixel là một hàng và RGB là cột
pixels = image.reshape(-1, 3)

# Thử phân cụm với các giá trị k khác nhau
k_values = [2, 3, 4, 5]
for k in k_values:
    # Khởi tạo và huấn luyện mô hình KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    # Lấy nhãn cụm cho mỗi pixel và tạo ảnh phân cụm
    segmented_img = kmeans.labels_.reshape(image.shape[:2])

    # Tính silhouette score để đánh giá phân cụm
    score = silhouette_score(pixels, kmeans.labels_)
    print(f'Silhouette Score cho k={k}: {score}')

    # Hiển thị ảnh đã phân cụm
    plt.figure(figsize=(8, 6))
    plt.imshow(segmented_img, cmap='viridis')
    plt.title(f'Phân cụm KMeans với k={k}')
    plt.axis('off')
    plt.show()

    # Lưu ảnh phân cụm nếu cần thiết
    io.imsave(f'clustered_image_k{k}.jpg', segmented_img)

