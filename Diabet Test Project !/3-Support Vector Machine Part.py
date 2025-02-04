from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# بارگذاری مجموعه داده
import pandas as pd
data = pd.read_csv('cleaned_diabetes_prediction_dataset.csv')

# کدگذاری متغیرهای دسته‌بندی  به مقدار عددی
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# تقسیم داده‌ها به ویژگی‌ها x  و برچسب هدف y
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# نرمال‌سازی ویژگی‌ها برای استفاده در SVM
scaler = StandardScaler()
X = scaler.fit_transform(X)

# تقسیم مجموعه داده به داده‌های آموزش و آزمایش
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# مقداردهی اولیه و آموزش مدل SVM
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# انجام پیش‌بینی‌ها
y_pred = svm_model.predict(X_test)

# ارزیابی مدل
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("دقت مدل:", accuracy)
print("گزارش طبقه‌بندی:\n", report)