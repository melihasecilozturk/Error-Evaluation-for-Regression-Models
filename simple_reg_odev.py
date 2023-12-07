import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score


df = pd.read_excel("/Users/melihasecilozturk/Desktop/miuul/ödevler/simple_reg_odev/denem.xlsx")
df.head()

# 1-Verilen bias ve weight’e göre doğrusal regresyon model denklemini oluşturunuz.
#Bias = 275, Weight= 90 (y’ = b+wx)

275 + (90 * df["deneyim"])



# 2-Oluşturduğunuz model denklemine göre tablodaki tüm deneyim yılları için maaş tahmini yapınız.

y_pred = 275 + (90 * df["deneyim"])


# 3-Modelin başarısını ölçmek için MSE, RMSE, MAE skorlarını hesaplayınız

df["y_pred"] = 275 + (90 * df["deneyim"])
df["hata"] = df["maas"]-df["y_pred"]
df["hata_kare"] = df["hata"]**2
df["mtlak_hata"] = abs(df["hata"])

#MSE

df["hata_kare"].mean()
#4438.333333333333


#RMSE

(df["hata_kare"].mean())**0.5

#MAE

df["mtlak_hata"].mean()



