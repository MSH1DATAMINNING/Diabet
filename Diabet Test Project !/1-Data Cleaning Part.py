import pandas as pd #  M.sh
import numpy as np # گذارش کار توضیحات تکمیلی هست در

# خواندن فایل CSV
file_path = "modified_diabetes_prediction_dataset.csv"  # مسیر فایل ورودی
df = pd.read_csv(file_path)

# تبدیل همه داده‌های متنی به حروف کوچک برای یکسان‌سازی
df = df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)

# جایگزینی مقدار ".9999" با NaN
df.replace(".9999", np.nan, inplace=True)

# حذف ردیف‌هایی که شامل مقدار "unknown" هستند
df = df[~df.apply(lambda row: row.astype(str).str.contains('unknown')).any(axis=1)]

#  جایگزینی مقدار "noinfo" با NaN
df.replace("noinfo", np.nan, inplace=True)

#  تبدیل ستون‌های عددی که ممکن است به اشتباه متنی ذخیره شده باشند
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# حذف ستون‌هایی که درصد زیادی مقدار NaN دارند (مثلاً بالای 50%)
threshold = 0.5
df = df.dropna(axis=1, thresh=len(df) * threshold)

#  حذف ردیف‌هایی که مقدار NaN زیادی دارند (مثلاً بیش از 30% از مقدارهایشان NaN باشد)
row_threshold = 0.3
df = df.dropna(axis=0, thresh=len(df.columns) * (1 - row_threshold))

#  پر کردن مقدارهای NaN باقی‌مانده با مقدار میانگین یا مد (بسته به نوع داده)
for col in df.columns:
    if df[col].dtype == 'float64' or df[col].dtype == 'int64':  # برای داده‌های عددی
        df[col].fillna(df[col].median(), inplace=True)
    else:  #برای داده‌های متنی
        df[col].fillna(df[col].mode()[0], inplace=True)

#  حذف مقادیر پرت با روش 
def remove_outliers(data, col):
    if data[col].dtype == 'float64' or data[col].dtype == 'int64':
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

for col in df.columns:
    df = remove_outliers(df, col)

# ذخیره داده‌های پاکسازی‌شده
cleaned_file_path = "cleaned_diabetes_prediction_dataset.csv"
df.to_csv(cleaned_file_path, index=False)
print(" داده‌های پاکسازی‌شده ذخیره شدند: {cleaned_file_path}")