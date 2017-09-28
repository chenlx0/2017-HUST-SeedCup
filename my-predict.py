# env /usr/bin/python3

import pandas as pd
import math
import csv
import random
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

# sign team elo
base_elo = 1600
team_elos = {}
team_stats = {}
X = []
y = []


# return team elo
def get_elos(team):
    try:
        return team_elos[team]
    except Exception as e:
        team_elos[team] = base_elo
        return team_elos[team]


# add team score, initializing tem stats
def add_score(team, score):
    try:
        team_stats[team] += score
    except KeyError as e:
        team_stats[team] = score


# initialize data
def init_data(team_data, person_data):

    # initialize team points data
    pattern = r'([0-9]+):([0-9]+)'
    team_data['比分'] = team_data['比分'].str.findall(pattern)
    team_data['比分'] = team_data['比分'].str.get(0)
    team_data['客队分数'] = team_data['比分'].str.get(0)
    team_data['主队分数'] = team_data['比分'].str.get(1)
    team_data = team_data.drop(["比分"], axis=1)

    person_data['评价'] = person_data['得分'] + 0.4 * person_data['投篮命中次数'] + \
        (-1) * 0.7 * person_data['投篮出手次数'] - 0.4 * (person_data['罚球出手次数'] - person_data['罚球命中次数']) + \
        0.7 * person_data['前场篮板'] + 0.3 * person_data['后场篮板'] + \
        person_data['抢断'] + 0.7 * person_data['助攻'] + \
        0.7 * person_data['盖帽'] - 0.4 * person_data['犯规'] - person_data['失误']

    person_data['评价'] = person_data['评价'] * 12

    for index, row in person_data.iterrows():
        add_score(row['队名'], row['评价'])

    return team_data


def calc_elo(win_team, lose_team, win_point, lose_point):
    winner_rank = get_elos(win_team)
    loser_rank = get_elos(lose_team)

    # calculate rank diff!
    rank_diff = winner_rank - loser_rank # + (win_point - lose_point)
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    # 根据rank级别修改K值
    if winner_rank < 2100:
        k = 32
    elif 2100 <= winner_rank < 2400:
        k = 24
    else:
        k = 16
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff

    return new_winner_rank, new_loser_rank


def build_data_set(all_data):
    print("Building data set..")
    X = []
    skip = 0
    for index, row in all_data.iterrows():

        home_name = row['主场队名']
        ano_name = row['客场队名']
        home_point = float(row['主队分数'])
        ano_point = float(row['客队分数'])

        if home_point > ano_point:
            Wteam = home_name
            Lteam = ano_name
            Wpoint = home_point
            Lpoint = ano_point
        else:
            Wteam = ano_name
            Lteam = home_name
            Wpoint = ano_point
            Lpoint = home_point

        #获取最初的elo或是每个队伍最初的elo值
        team1_elo = get_elos(Wteam)
        team2_elo = get_elos(Lteam)

        # 给主场比赛的队伍加上100的elo值
        if Wteam == home_name:
            team1_elo += 100
        else:
            team2_elo += 100

        # 把elo当为评价每个队伍的第一个特征值
        team1_features = [team1_elo]
        team2_features = [team2_elo]

        # 添加我们从basketball reference.com获得的每个队伍的统计信息

        # for key, value in team_stats.loc[Wteam].iteritems():
        #     team1_features.append(value)
        # for key, value in team_stats.loc[Lteam].iteritems():
        #     team2_features.append(value)

        team1_features.append(team_stats[Wteam])
        team2_features.append(team_stats[Lteam])
        # 将两支队伍的特征值随机的分配在每场比赛数据的左右两侧
        # 并将对应的0/1赋给y值
        if random.random() > 0.5:
            X.append(team1_features + team2_features)
            y.append(0)
        else:
            X.append(team2_features + team1_features)
            y.append(1)

        # 根据这场比赛的数据更新队伍的elo值
        new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam, Wpoint, Lpoint)
        team_elos[Wteam] = new_winner_rank
        team_elos[Lteam] = new_loser_rank

    return np.nan_to_num(X), y


def predict_winner(team_1, team_2, model):
    features = []

    # team 1，客场队伍
    features.append(get_elos(team_1))
    # for key, value in team_stats.loc[team_1].iteritems():
    #     features.append(value)

    features.append(team_stats[team_1])

    # team 2，主场队伍
    features.append(get_elos(team_2) + 100)
    # for key, value in team_stats.loc[team_2].iteritems():
    #     features.append(value)
    features.append(team_stats[team_2])

    features = np.nan_to_num(features)
    return model.predict_proba([features])


if __name__ == "__main__":
    team_data = pd.read_csv('matchDataTrain.csv')
    person_data = pd.read_csv('teamData.csv')

    preproccess_data = init_data(team_data, person_data)

    x_data, y_data = build_data_set(preproccess_data)

    print(x_data)

    print("Training . . . . .")

    model = linear_model.LogisticRegression()
    model.fit(x_data, y_data)

    # 利用10折交叉验证计算训练正确率
    print("Doing cross-validation..")
    print(cross_val_score(model, x_data, y_data, cv=10, scoring='accuracy', n_jobs=-1).mean())

    res = []
    test = pd.read_csv('matchDataTest.csv')
    k = 0
    while k < 911:
        i = test['主场队名'][k]
        j = test['客场队名'][k]
        k += 1
        res.append([float(predict_winner(j, i, model=model)[0][1])])

    with open('predictPro.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['主场赢得比赛的置信度'])
        writer.writerows(res)
