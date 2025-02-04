#hi
#analiz code ; by M.Sh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering

# 1. خواندن داده‌ها
file_path = "cleaned_diabetes_prediction_dataset.csv"  # مسیر فایل را تنظیم کنید
df = pd.read_csv(file_path)

# 2. پیش‌پردازش داده‌ها
df['gender'] = LabelEncoder().fit_transform(df['gender'])
df['smoking_history'] = LabelEncoder().fit_transform(df['smoking_history'])
df = df.drop(columns=['diabetes'])  # حذف برچسب

# 3. استانداردسازی داده‌ها
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 4. رسم دندروگرام برای تعیین تعداد بهینه خوشه‌ها
plt.figure(figsize=(10, 5))
dendrogram = sch.dendrogram(sch.linkage(df_scaled, method='ward'))
plt.title("دندروگرام برای خوشه‌بندی سلسله‌مراتبی")
plt.xlabel("نمونه‌ها")
plt.ylabel("فاصله خوشه‌ها")
plt.show()

# 5. اجرای خوشه‌بندی سلسله‌مراتبی با تعداد خوشه انتخابی (مثلاً 3)
optimal_k = 3  # تعداد خوشه بهینه را از دندروگرام انتخاب کنید
hc = AgglomerativeClustering(n_clusters=optimal_k, affinity='euclidean', linkage='ward')
df['cluster'] = hc.fit_predict(df_scaled)

# نمایش تعداد نمونه‌های هر خوشه
print(df['cluster'].value_counts())

# 6. تجسم خوشه‌بندی با PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
df_pca['cluster'] = df['cluster']

# رسم نمودار خوشه‌بندی
plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    plt.scatter(df_pca[df_pca['cluster'] == cluster]['PCA1'], 
                df_pca[df_pca['cluster'] == cluster]['PCA2'], label=f'خوشه {cluster}')

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('نمایش خوشه‌های Hierarchical Clustering با کاهش ابعاد PCA')
plt.legend()
plt.show()