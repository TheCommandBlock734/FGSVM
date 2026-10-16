import ast
import math
import random
import time as systime
import warnings

import numpy
import numpy as np
from sklearn.cluster import k_means
from sklearn.metrics.pairwise import euclidean_distances

# 1.输入数据data
# 2.打印绘制原始数据
# 3.判断粒球的纯度
# 4.纯度不满足要求，k-means划分粒球
# 5.绘制每个粒球的数据点
# 6.计算粒球均值，得到粒球中心和半径，绘制粒球


# Calculate label and purity of granular-balls
def get_label_and_purity(gb):
    # Calculate the number of data categories
    len_label = numpy.unique(gb[:, 0], axis=0)
    # print(len_label)

    if len(len_label) == 1:
        purity = 1.0
        label = len_label[0]
    else:
        num = gb.shape[0]
        gb_label_temp = {}
        for label in len_label.tolist():
            # Separate data with different labels
            gb_label_temp[sum(gb[:, 0] == label)] = label
        # print(gb_label_temp)
        # The proportion of the largest category of data in all data
        max_label = max(gb_label_temp.keys())
        purity = max_label / num if num else 1.0
        label = gb_label_temp[max_label]
    # print(label)
    # label, purity
    return label, purity


# Calculate granular-balls center and radius
def calculate_center_and_radius(gb):
    data_no_label = gb[:, 1:]
    # print(data_no_label)
    center = data_no_label.mean(axis=0)
    radius_mean = numpy.mean((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    # radius_max = numpy.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius_mean


# Calculate distance
def calculate_distances(data, p):
    # print(data, p)
    return ((data - p) ** 2).sum(axis=0) ** 0.5


def splits(purity, gb_dict):
    gb_dict_new = {}
    while True:
        # Copy a temporary list, and then traverse the value
        if len(gb_dict_new) == 0:
            # initial assignment
            gb_dict_temp = gb_dict.copy()
        else:
            # Subsequent traversal assignment
            gb_dict_temp = gb_dict_new.copy()
        gb_dict_new = {}
        # record the number before a division
        ball_number_1 = len(gb_dict_temp)
        # print("ball_number_1:", ball_number_1)
        for key in gb_dict_temp.keys():
            gb_single = {}
            # Fetch dictionary data, including keys and values
            gb_single[key] = gb_dict_temp[key]
            # print("gb_single:", gb_single)
            gb = gb_single[key][0]
            # print(len(gb))
            if len(gb) == 0:
                continue

            # purity
            p = get_label_and_purity(gb)[1]
            # print(p)
            # Determine whether the purity meets the requirements, if not, continue to divide
            if p < purity and len(gb) != 1:
                # print(gb_single)
                gb_dict_re = splits_ball(gb_single).copy()
                gb_dict_new.update(gb_dict_re)
            else:
                gb_dict_new.update(gb_single)
                continue
        # gb_dict_new = adjust_center(gb_dict_new)
        # record the number after a division
        ball_number_2 = len(gb_dict_new)
        # print("ball_number_2:", len(gb_dict_new))
        # The number of granular-balls is the same as the number of granular-balls last divided, that is, it will not change
        if ball_number_1 == ball_number_2:
            break

        # draw granular-balls
        # gb_plot(gb_dict_new)

    # draw granular-balls
    # gb_plot(gb_dict_new)
    return gb_dict_new


# split granular-balls
def splits_ball(gb_dict):
    # {center: [gb, distances]}
    center = []  # old center
    distances_other_class = []  # distance to heterogeneous data points
    balls = []  # The result after clustering
    center_other_class = []
    ball_list = (
        {}
    )  # Returned dictionary result, key: center point, value (ball , distance from data to center)
    distances_other_temp = []

    centers_dict = []  # centers
    gbs_dict = []  # data
    distances_dict = []  # distances

    # Fetch dictionary data, including keys and values
    gb_dict_temp = gb_dict.popitem()
    for center_split in gb_dict_temp[0].split("_"):
        center.append(float(center_split))
    center = np.array(center)
    gb = gb_dict_temp[1][0]  # Get granular-ball data
    distances = gb_dict_temp[1][1]  # Take out the distance to the old center
    # print('center:', center)
    # print('gb:', gb)
    # print('distances:', distances)
    centers_dict.append(center)  # old center join the centers

    # Take a new center
    len_label = numpy.unique(gb[:, 0], axis=0)
    # When the input has only one type of data, select a point different from the original center
    if len(len_label) > 1:
        gb_class = len(len_label)
    else:
        gb_class = 2
    # Take multiple centers for multiple types of data
    len_label = len_label.tolist()
    for i in range(0, gb_class - 1):
        # print(len_label)
        if len(len_label) < 2:
            # When de-overlapping, there is no heterogeneous point situation
            gb_temp = np.delete(gb, distances.index(0), axis=0)  # Remove the old center
            ran = random.randint(0, len(gb_temp) - 1)
            center_other_temp = gb_temp[ran]  # Take a new center
            center_other_class.append(center_other_temp)
        else:
            if center[0] in len_label:
                len_label.remove(center[0])
            gb_temp = gb[gb[:, 0] == len_label[i], :]  # Extract heterogeneous data

            # random center of heterogeneity
            ran = random.randint(0, len(gb_temp) - 1)
            center_other_temp = gb_temp[ran]

            # center_other_temp = select_center(gb_temp)
            # print(center_other_temp)
            center_other_class.append(center_other_temp)
            # print(distances.index(max(distances)))
    # print('center_other_class:', center_other_class)
    # join the centers
    centers_dict.extend(center_other_class)
    # print('centers_dict:', centers_dict)

    # Store all data distance to old center
    distances_other_class.append(distances)
    # Calculate the distance to each new center
    for center_other in center_other_class:
        balls = []  # The result after clustering
        distances_other = []
        for feature in gb:
            distances_other.append(calculate_distances(feature[1:], center_other[1:]))
        # new centers
        # distances_dict.append(distances_other)
        distances_other_temp.append(
            distances_other
        )  # Temporary storage distance to each new center
        # Store all data distance to new center
        distances_other_class.append(distances_other)
    # print('distances_other_class:', len(distances_other_class))

    # The distance from a certain data to the original center and the new center, take the smallest for classification
    for i in range(len(distances)):
        distances_temp = []
        distances_temp.append(distances[i])
        for distances_other in distances_other_temp:
            distances_temp.append(distances_other[i])
        # print('distances_temp:', distances_temp)
        classification = distances_temp.index(
            min(distances_temp)
        )  # 0:old center；1,2...：new centers
        balls.append(classification)
    # Clustering situation
    balls_array = np.array(balls)
    # print("Clustering situation：", balls_array)

    # Assign data based on clustering
    for i in range(0, len(centers_dict)):
        gbs_dict.append(gb[balls_array == i, :])
    # print('gbs_dict:', gbs_dict)

    # assign new distance
    i = 0
    for j in range(len(centers_dict)):
        distances_dict.append([])
    # print('distances_dict:', distances_dict)
    for label in balls:
        distances_dict[label].append(distances_other_class[label][i])
        i += 1
    # print('distances_dict:', distances_dict)

    # packed into a dictionary
    for i in range(len(centers_dict)):
        gb_dict_key = str(float(centers_dict[i][0]))
        for j in range(1, len(centers_dict[i])):
            gb_dict_key += "_" + str(float(centers_dict[i][j]))
        gb_dict_value = [
            gbs_dict[i],
            distances_dict[i],
        ]  # Pellets + distance to centers
        ball_list[gb_dict_key] = gb_dict_value

    # print('ball_list:', ball_list)
    return ball_list


def toGB(X, y, purity):
    data = np.hstack([y.reshape(-1, 1), X])
    warnings.filterwarnings("ignore")  # ignore warning

    start = systime.time()
    ball_list = []

    # initial random center
    center_init = data[random.randint(0, len(data) - 1), :]
    # center_init = data[:, 1:3].mean(axis=0)
    # print(center_init)

    # The distance from the initial center to the data
    distance_init = []
    for feature in data:
        distance_init.append(calculate_distances(feature[1:], center_init[1:]))
    # print('distance_init:', len(distance_init))

    # packed into a dictionary:{center: [gb, distances]}
    gb_dict = {}
    gb_dict_key = str(center_init.tolist()[0])
    for j in range(1, len(center_init)):
        gb_dict_key += "_" + str(center_init.tolist()[j])
    gb_dict_value = [data, distance_init]
    gb_dict[gb_dict_key] = gb_dict_value

    # first time drawing a sphere
    # gb_plot2(gb_dict)

    # Classification
    gb_dict = splits(purity=purity, gb_dict=gb_dict)

    k_centers = []
    splits_k = len(gb_dict)
    for key in gb_dict.keys():
        k_center = []
        for k in key.split("_"):
            k_center.append(float(k))
        k_centers.append(k_center[1:])
    # print(np.array(k_centers))
    # Perform a global division
    label_cluster = k_means(
        X=data[:, 1:],
        n_clusters=splits_k,
        n_init=2,
        init=np.array(k_centers),
        random_state=42,
    )[1]
    for single_label in range(splits_k):
        ball_list.append(data[label_cluster == single_label, :])
    end = systime.time()

    C = []
    r = []
    y = []
    n = []
    p = []
    for gb in ball_list:
        num = gb.shape[0]
        center = np.mean(gb[:, 1:], axis=0)
        radius = np.mean(np.linalg.norm(gb[:, 1:] - center, axis=1))
        labels, counts = np.unique(gb[:, 0], return_counts=True)
        C.append(center)
        r.append(radius)
        y.append(labels[counts.argmax()])
        n.append(num)
        p.append(counts.max() / num)
    C = np.stack(C)
    r = np.array(r)
    y = np.array(y, dtype=int)
    n = np.array(n, dtype=int)
    p = np.array(p)
    return end - start, C, r, y, n, p, ball_list
