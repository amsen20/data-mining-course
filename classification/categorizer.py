import pandas as pd

int_data = {}
numerals = {'float64', 'int64'}


def extract(path):
    df = pd.read_csv(path)
    for col in df.columns:
        if str(df[col].dtype) in numerals:
            int_data[col] = []
            for it in df[col]:
                int_data[col].append(it)


def prep(dataset_id):
    extract(f"Dataset/Dataset{dataset_id}.csv")
    extract(f"Dataset/Dataset{dataset_id}_Unknown.csv")


categorizers = {}


def calc_categorizers():
    for col in int_data:
        int_data[col].sort()
        n = len(int_data[col])
        fp = int_data[col][n//3]
        sp = int_data[col][2*n//3]

        def gen(fp, sp):
            def f(x):
                if x <= fp:
                    return 'A'
                if x <= sp:
                    return 'B'
                return 'C'
            return f

        categorizers[col] = gen(fp, sp)


def get_cat_data(path):
    data = {}
    df = pd.read_csv(path)
    for col in df.columns:
        data[col] = []
        if str(df[col].dtype) in numerals:
            for it in df[col]:
                data[col].append(categorizers[col](it))
        else:
            for it in df[col]:
                data[col].append(it)
    return data
