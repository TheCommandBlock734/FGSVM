import random
import signal
import time as systime

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm
from ucimlrepo import fetch_ucirepo

from classifier import FGSVM, GBSVM
from generator import GBG, GBG_plusplus


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException()


def run_with_timeout(func, timeout, *args, **kwargs):
    handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


def warm_up(size):
    A = np.random.random((size, size)) + np.eye(size)
    return (np.linalg.inv(A.T @ A) @ A).sum()


def get_UCI(id):
    print(f"Fetching id={id}")
    repo = fetch_ucirepo(id=id)
    df = repo.data.features
    dup_cols = df.T.duplicated(keep="last")
    df = df.loc[:, ~dup_cols]
    mask = ~df.duplicated(keep="last")
    x, y = df[mask], repo.data.targets[mask]

    num = x.select_dtypes(include=[np.number]).columns.tolist()
    cat = x.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(), cat),
        ]
    )

    label_encoder = LabelEncoder()

    x = preprocessor.fit_transform(x)
    if issparse(x):
        x = x.toarray()
    y = label_encoder.fit_transform(y.values.ravel())
    return x, y


def add_noise(y, ratio):
    y = y.copy()
    num = int(ratio * len(y))
    idx = np.random.choice(y.shape[0], size=num, replace=False)
    y[idx] = 1 - y[idx]
    return y


def one_vs_one(experiments, dataset, features, targets):
    n = targets.max()
    for i in range(n):
        for j in range(i + 1, n + 1):
            idx = (targets == i) | (targets == j)
            X = features[idx]
            y = targets[idx]
            y = np.where(y == i, 0, 1)
            experiments.append((f"{dataset}:{i}vs{j}", X, y))


def compare_algorithms(
    pbar,
    timeout,
    compare,
    ablation,
    dataset,
    noise_level,
    split,
    generator,
    gen_time,
    C,
    r,
    y,
    points,
):
    best_FG_time = timeout
    best_FG_acc = best_FG_rec = best_FG_C = 0
    best_FG_solver = best_FG_b = best_FG_w = None
    for C_xi in [2**i for i in range(-7, 8)]:
        for solver in [cp.CLARABEL, cp.ECOS, cp.MOSEK]:
            pbar.set_postfix(
                method="FGSVM",
                generator=generator,
                dataset=dataset,
                split=split,
                noise_level=noise_level,
                C=C_xi,
                solver=solver,
            )
            try:
                time, acc, rec, w, b, support, tightness = run_with_timeout(
                    FGSVM.runClassifier,
                    timeout,
                    C,
                    r,
                    y,
                    tX,
                    ty,
                    C_xi=C_xi,
                    solver=solver,
                )
                if acc > best_FG_acc or (
                    acc + 1e-5 > best_FG_acc and time > best_FG_time
                ):
                    best_FG_time = time
                    best_FG_acc = acc
                    best_FG_rec = rec
                    best_FG_C = C_xi
                    best_FG_solver = solver
                    best_FG_b = b
                    best_FG_w = w
                ablation.loc[len(ablation)] = [
                    dataset,
                    noise_level,
                    split,
                    generator,
                    solver,
                    C_xi,
                    "done",
                    time,
                    acc,
                    rec,
                    support,
                    tightness,
                ]
            except TimeoutException:
                ablation.loc[len(ablation)] = [
                    dataset,
                    noise_level,
                    split,
                    generator,
                    solver,
                    C_xi,
                    "timeout",
                    timeout,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            except cp.error.SolverError:
                ablation.loc[len(ablation)] = [
                    dataset,
                    noise_level,
                    split,
                    generator,
                    solver,
                    C_xi,
                    "failed",
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]

    best_ori_time = timeout
    best_ori_acc = best_ori_rec = best_ori_C = best_ori_iter = 0
    best_ori_b = best_ori_w = None
    for C_xi in [2**i for i in range(-7, 8)]:
        pbar.set_postfix(
            method="GBSVM",
            generator=generator,
            dataset=dataset,
            split=split,
            noise_level=noise_level,
            C=C_xi,
        )
        try:
            time, acc, rec, w, b, count = run_with_timeout(
                GBSVM.runClassifier,
                timeout,
                C,
                r,
                y,
                points,
                tX,
                ty,
                max_iter=2000,
                C_xi=C_xi,
            )
            if acc > best_ori_acc:
                best_ori_time = time
                best_ori_acc = acc
                best_ori_rec = rec
                best_ori_C = C_xi
                best_ori_iter = count
                best_ori_b = b
                best_ori_w = w
        except TimeoutException:
            pass

    if best_ori_w is None:
        compare.loc[len(compare)] = (
            [dataset, noise_level, split, generator, gen_time]
            + [np.nan for i in range(7)]
            + [
                best_FG_solver,
                best_FG_C,
                best_FG_time,
                best_FG_acc,
                best_FG_rec,
                best_FG_b,
                np.linalg.norm(best_FG_w),
            ]
        )
    else:
        compare.loc[len(compare)] = [
            dataset,
            noise_level,
            split,
            generator,
            gen_time,
            best_ori_C,
            best_ori_time,
            best_ori_acc,
            best_ori_rec,
            best_ori_iter,
            best_ori_b,
            np.linalg.norm(best_ori_w),
            best_FG_solver,
            best_FG_C,
            best_FG_time,
            best_FG_acc,
            best_FG_rec,
            best_FG_b,
            np.linalg.norm(best_FG_w),
        ]

experiments = []

x, y = get_UCI(451)
experiments.append(("breast_cancer_coimbra", x, y))

x, y = get_UCI(17)
experiments.append(("breast_cancer_wisconsin_diagnostic", x, y))

x, y = get_UCI(53)
one_vs_one(experiments, "iris", x, y)

x, y = get_UCI(109)
one_vs_one(experiments, "wine", x, y)

x, y = get_UCI(728)
experiments.append(("qsar_toxicity", x, y))

x, y = get_UCI(292)
one_vs_one(experiments, "wholesale_customers", x, y)

x, y = get_UCI(147)
one_vs_one(experiments, "image_segmentation", x, y)

x, y = get_UCI(850)
experiments.append(("raisin", x, y))

x, y = get_UCI(267)
experiments.append(("banknote_authentication", x, y))

x, y = get_UCI(146)
one_vs_one(experiments, "landsat_satellite", x, y)

x, y = get_UCI(545)
experiments.append(("rice_cammeo_and_osmancik", x, y))

x, y = get_UCI(602)
one_vs_one(experiments, "dry_bean_dataset", x, y)

x, y = get_UCI(372)
experiments.append(("htru2", x, y))

x, y = get_UCI(229)
experiments.append(("skin_segmentation", x, y))

compare_col = (
    [
        "dataset",
        "noise_level",
        "split",
        "generator",
        "gen_time",
    ]
    + [f"ori_{p}" for p in ["C", "time", "accuracy", "recall", "iter", "b"]]
    + ["||ori_w||"]
    + [f"FG_{p}" for p in ["solver", "C", "time", "accuracy", "recall", "b"]]
    + ["||FG_w||"]
)

ablation_col = [
    "dataset",
    "noise_level",
    "split",
    "generator",
    "solver",
    "C",
    "status",
    "time",
    "accuracy",
    "recall",
    "num_support",
    "tightness",
]

compare = pd.DataFrame(columns=compare_col)
ablation = pd.DataFrame(columns=ablation_col)
compare.to_csv("./compare.csv", index=False)
ablation.to_csv("./ablation.csv", index=False)

print("Warming up:")
print(warm_up(10000))

pbar = tqdm(experiments, desc="Running algorithms")
for e in pbar:
    compare = pd.DataFrame(columns=compare_col)
    ablation = pd.DataFrame(columns=ablation_col)
    dataset, data, label = e
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for split, (train, test) in zip(range(5), skf.split(data, label)):
        X, l = data[train], label[train]
        tX, ty = data[test], label[test]
        for noise in range(0, 5):
            noise_level = noise * 0.05
            nl = add_noise(l, noise_level)
            np.random.seed(42)
            random.seed(42)
            try:
                gen_time, C, r, y, n, p, points = GBG.toGB(X, nl, 0.95)
            except:
                pass
            else:
                compare_algorithms(
                    pbar,
                    1,
                    compare,
                    ablation,
                    dataset,
                    noise_level,
                    split,
                    "GBG",
                    gen_time,
                    C,
                    r,
                    y,
                    points,
                )
            try:
                gen_time, C, r, y, n, p, points = GBG_plusplus.toGB(X, nl, 0.95)
            except:
                pass
            else:
                compare_algorithms(
                    pbar,
                    1,
                    compare,
                    ablation,
                    dataset,
                    noise_level,
                    split,
                    "GBG++",
                    gen_time,
                    C,
                    r,
                    y,
                    points,
                )
    compare.to_csv("./compare.csv", index=False, mode="a", header=False)
    ablation.to_csv("./ablation.csv", index=False, mode="a", header=False)
print("All done.")