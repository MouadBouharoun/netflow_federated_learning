
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



def preprocess(df,attack_mapping):
    src_ipv4_idx = {name: idx for idx, name in enumerate(sorted(df["IPV4_SRC_ADDR"].unique()))}
    dst_ipv4_idx = {name: idx for idx, name in enumerate(sorted(df["IPV4_DST_ADDR"].unique()))}
    attack_idx = {name: idx for idx, name in enumerate(sorted(df["Attack"].unique()))}

    df["IPV4_SRC_ADDR"] = df["IPV4_SRC_ADDR"].apply(lambda name: src_ipv4_idx[name])
    df["IPV4_DST_ADDR"] = df["IPV4_DST_ADDR"].apply(lambda name: dst_ipv4_idx[name])
    df['Attack'] = df['Attack'].map(attack_mapping)

    X = df.iloc[:, :-1].values
    y = df["Attack"].values

    # Normaliser les données
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    return (X_train, X_test, y_train, y_test)
