# Emirhan Temizkol
# Problem
# Bir e-ticaret sitesindeki belirli bir ürünün yorumlarının ve değerlendirmelerinin bulunduğu veriseti bulunmakta
# 1. Amacımız ürünün son zamanlardaki beğeni trendine ağırlık vererek puanını hesaplamak.
# 2. Amacımız akla ve mantığa uygun şekilde bir puanlama sistemi oluşturarak en faydali ilk 20 yorumu belirlemek.

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.expand_frame_repr', False)

df_ = pd.read_csv(r"D:\DATA SCIENCE\VAHİT BAŞKAN\6. Hafta\Sorting\df_sub.csv")
df = df_.copy()
df.head()
df.info()
df.shape

# Görev 1

df["overall"].mean()
# Ürünün ortalama puanı : 4.58

# reviewTime için datetime dönüşümü
df["reviewTime"] = pd.to_datetime(df["reviewTime"], dayfirst=True)
df["reviewTime"].max()
# En son yorumdan bir sonraki günü analiz yaptığımız tarih olarak tanımlayalım
current_date = pd.to_datetime("2014-12-08 0:0:0")
df.head()

df["day_diff"] = (current_date-df["reviewTime"]).dt.days
# Yakın tarihten uzaktarihe A,B,C,D [A yakın, D uzak]
df["day_diff_segment"] = pd.qcut(df["day_diff"],q=[0,.25,.50,.75,1],labels=["A","B","C","D"])
df.head()
# Tarih ağırlıklı ortalama,
# df.loc[satır,sütun]

date_weighted_mean = df.loc[df["day_diff_segment"]=="A","overall"].mean() * 28/100 + \
    df.loc[df["day_diff_segment"]=="B","overall"].mean() * 26/100 + \
    df.loc[df["day_diff_segment"]=="C","overall"].mean() * 24/100 + \
    df.loc[df["day_diff_segment"]=="D","overall"].mean() * 22/100
print(date_weighted_mean)

# Tarih ağırlıklı ürün ortalaması:
# 4.595593165128118
# Aritmetik ortalama:
# 4.58

# İki ortalamayı kıyasladığımızda ürünün son dönemde biraz daha beğenildiği ve yüksek puanlar aldığı gözükmektedir.

# Görev 2

df.head(100)
df.info()
type(df.helpful[0])

df["helpful_edit"] = df.helpful

import json
# string halinde verilen listeyi tekrardan listeye dönüştürebilmek
# için json.loads(str_list) kullanılabilir.

df["helpful_yes"] = df["helpful_edit"].apply(lambda x: json.loads(x)[0])
df["helpful_total"] = df["helpful_edit"].apply(lambda x: json.loads(x)[1])
df["helpful_no"] = df["helpful_total"] - df["helpful_yes"]

# pos - neg score
df["score_pos_neg_diff"] = df["helpful_yes"] - df["helpful_no"]

# average score
df["score_average_rating"] = df["helpful_yes"] / df["helpful_total"]

# wilson lower bound score

def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not: Eğer skorlar 1-5 arasıdaysa 1-3 down, 4-5 up olarak işaretlenir ve bernoulli'ye uygun hale getirilir.

    Parameters
    ----------
    pos: int
        pozitif yorum sayısı
    neg: int
        negatif yorum sayısı
    confidence: float
        güven aralığı

    Returns
    -------
    wilson score: float

    """
    import math
    import scipy.stats as st
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["score_wlb"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Bu üç skor, ağırlıkları belirlenerek çarpılır ve genel skor bulunur.
# Bu skora göre büyükten küçüğe sıraladığımız zaman ilk 20 yorum listelenmiş olur.
df["score_average_rating"].fillna(0,inplace = True)
df["score_total"] = df["score_wlb"] * 40/100 + df["score_average_rating"] * 30/100 + df["score_pos_neg_diff"] * 30/100

df.head()
# %40 wilson lower bound
# %30 positif negatif diff (positive - negative)
# %30 average rating (positive/total)
# Skorlarından oluşan total skorumuz için sıralanmış 20 yorum aşağıdaki gibidir.
df.sort_values("score_total",ascending=False).head(20)