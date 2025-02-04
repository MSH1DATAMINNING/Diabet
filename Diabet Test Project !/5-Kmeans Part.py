import pandas as pd
import numpy as np # گذارش کار هم در باره کد و داده ها موجود میباشد در pdf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

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

# 4. انتخاب تعداد بهینه خوشه‌ها با روش Elbow
inertia = []
K_range = range(1, 11)  # بررسی تعداد خوشه‌ها از 1 تا 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# رسم نمودار Elbow
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel('تعداد خوشه‌ها (K)')
plt.ylabel('Inertia (مجموع مربعات فواصل)')
plt.title('روش Elbow برای انتخاب تعداد خوشه‌ها')
plt.show()

# 5. اجرای K-Means با تعداد خوشه‌های انتخاب‌شده (مثلاً 3 خوشه)
optimal_k = 3  # مقدار K بهینه را بر اساس نمودار انتخاب کنید
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df_scaled)

# نمایش تعداد نمونه‌های هر خوشه
print(df['cluster'].value_counts())

# 6. تجسم خوشه‌بندی با استفاده از PCA (کاهش ابعاد به 2 بُعد)
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
plt.title('نمایش خوشه‌های K-Means با کاهش ابعاد PCA')
plt.legend()
plt.show()