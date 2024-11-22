import pandas as pd
from kmodes.kprototypes import KPrototypes
import streamlit as st
import numpy as np

# Tiêu đề ứng dụng
st.title("Phân cụm K-Prototypes với điểm GPA, tính cách và sở thích (cân bằng cụm)")

# Tải lên file CSV
uploaded_file = st.file_uploader("Tải lên file CSV của bạn", type=["csv"])

if uploaded_file is not None:
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv(uploaded_file)

    # Hiển thị dữ liệu ban đầu
    st.write("### Dữ liệu ban đầu:")
    st.write(data)

    # Tùy chỉnh số lượng cụm
    num_clusters = st.slider("Chọn số lượng cụm", min_value=2, max_value=10, value=3, step=1)

    # Xử lý giá trị trống
    data.fillna({
        "GPA (Hệ 10)": data["GPA (Hệ 10)"].mean(),
        "GPA kì gần nhất (Hệ 10)": data["GPA kì gần nhất (Hệ 10)"].mean(),
        "Điểm QTHT (điểm cuối cùng)": data["Điểm QTHT (điểm cuối cùng)"].mean(),
        "Tính cách": "Không rõ",
        "Sở thích": "Không rõ"
    }, inplace=True)

    # Chuyển đổi các cột điểm số thành kiểu số
    numeric_cols = ["GPA (Hệ 10)", "GPA kì gần nhất (Hệ 10)", "Điểm QTHT (điểm cuối cùng)"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Tính điểm trung bình của ba cột GPA
    data["Điểm trung bình"] = data[numeric_cols].mean(axis=1)

    # Gắn nhãn học lực dựa trên điểm trung bình
    def assign_level(avg_score):
        if avg_score >= 8.5:
            return "Giỏi"
        elif avg_score >= 7.0:
            return "Khá"
        elif avg_score >= 5.5:
            return "Trung bình"
        else:
            return "Yếu"

    data["Học lực"] = data["Điểm trung bình"].apply(assign_level)

    # Hiển thị dữ liệu sau xử lý
    st.write("### Dữ liệu sau xử lý:")
    st.write(data)

    # Lọc các cột cần thiết cho quá trình phân cụm
    clustering_data = data[numeric_cols + ["Tính cách", "Sở thích"]]
    categorical_cols = ["Tính cách", "Sở thích"]  # Các cột phân loại

    # Tìm vị trí cột phân loại
    categorical_indices = [clustering_data.columns.get_loc(col) for col in categorical_cols]

    # Chạy thuật toán K-Prototypes
    kproto = KPrototypes(n_clusters=num_clusters, init="Huang", random_state=42)
    clusters = kproto.fit_predict(clustering_data, categorical=categorical_indices)

    # Gắn nhãn cụm vào dữ liệu
    data["Cluster"] = clusters + 1

    # Phân bổ lại để các cụm đều nhau
    st.write("### Phân bổ lại các thành viên để các cụm cân bằng:")
    def balance_clusters(data, cluster_col="Cluster", num_clusters=num_clusters):
        # Đếm số lượng trong mỗi cụm
        cluster_counts = data[cluster_col].value_counts()
        st.write("Số lượng ban đầu trong mỗi cụm:")
        st.write(cluster_counts)

        # Tính số lượng lý tưởng mỗi cụm
        ideal_count = len(data) // num_clusters
        adjustments = []
        to_move_indices = []

        # Phân tích các cụm cần điều chỉnh
        for cluster, count in cluster_counts.items():
            if count > ideal_count:  # Cụm dư người
                excess = count - ideal_count
                excess_indices = data[data[cluster_col] == cluster].sample(excess).index
                to_move_indices.extend(excess_indices)
                adjustments.append((cluster, -excess))
            elif count < ideal_count:  # Cụm thiếu người
                deficit = ideal_count - count
                adjustments.append((cluster, deficit))

        # Thực hiện điều chỉnh
        for cluster, adjustment in adjustments:
            if adjustment > 0:  # Nếu cụm thiếu người
                available_indices = to_move_indices[:adjustment]
                data.loc[available_indices, cluster_col] = cluster
                to_move_indices = to_move_indices[adjustment:]

        return data

    data = balance_clusters(data)

    # Hiển thị kết quả phân cụm
    st.write("### Kết quả phân cụm sau cân bằng:")
    st.write(data)

    # Hiển thị dữ liệu theo từng cụm
    st.write("### Chia dữ liệu theo từng cụm:")
    for cluster in sorted(data["Cluster"].unique()):
        st.write(f"#### Cụm {int(cluster)}:")
        cluster_data = data[data["Cluster"] == cluster]
        st.write(cluster_data)

    # Hiển thị biểu đồ phân cụm
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.write("### Biểu đồ phân cụm:")
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=data,
        x="GPA (Hệ 10)",
        y="GPA kì gần nhất (Hệ 10)",
        hue="Cluster",
        palette="viridis",
        ax=ax
    )
    st.pyplot(fig)

else:
    st.info("Vui lòng tải lên file CSV để bắt đầu.")
