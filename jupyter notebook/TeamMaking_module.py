from random import choice
from simanneal import Annealer
from copy import deepcopy

tdfk = ['北海道', '青森', '岩手', '宮城', '秋田', '山形', '福島',
        '茨城', '栃木', '群馬', '埼玉', '千葉', '東京', '神奈川',
        '新潟', '富山', '石川', '福井', '山梨', '長野', '岐阜',
        '静岡', '愛知', '三重', '滋賀', '京都', '大阪', '兵庫',
        '奈良', '和歌山', '鳥取', '島根', '岡山', '広島', '山口', 
        '徳島', '香川', '愛媛', '高知', '福岡', '佐賀', '長崎',
        '熊本', '大分', '宮崎', '鹿児島', '沖縄', '海外', 'なし']

def sort_by_todofuken(data, tdfk=tdfk, by=['現住所', '出身', '氏名']):
    """
    現住所、出身地を北から都道府県をソート

    Args:
        data : pandas.DataFrame
        tdfk : list
        by : list

    Returns:
        [tdfk] と [by]によってソートされたDataFrame
    """
    for i in range(len(tdfk)):
        data['現住所'].replace(tdfk[i], i, inplace=True)
    data['現住所'].astype(int)

    data.sort_values(by=by, inplace=True)

    for i in range(len(tdfk)):
        data['現住所'].replace(i, tdfk[i], inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data

def sameTeam_count(sameTeam_num, teamList_T):
    """
    メンバー同士が同じチームに属した回数を記録

    Args:
        sameTeam_num : numpy.ndarray
            誰と何回同じチームになったかを記録する配列
        teamList_T : numpy.ndarray
            ある人がどのチームに属しているかを格納している配列
    
    Return:
        output : numpy.ndarray
            誰と何回同じチームになったかを記録する配列
    """
    output = deepcopy(sameTeam_num)
    for team in teamList_T:
        for member_i in range(len(team)):
            member_1 = team[member_i]
            for member_j in range(member_i+1, len(team)):
                member_2 = team[member_j]
                member_1 = int(member_1)
                member_2 = int(member_2)
                output[member_1][member_2] += 1
                output[member_2][member_1] += 1
    return output

def womenLess_count(womenLess_list, team_ravel, gender_count_list):
    """
    女性の数が少ないチームになった回数を記録

    Args:
        womenLess_list: numpy.array
            女性が少ないチームになった回数を格納している配列
        team_ravel: numpy.ndarray
            どのチームに所属しているかを格納している配列
        gender_count_list: dictionary
            チーム毎の性別の数を格納した辞書

    Return:
        output:numpy.array
            女性が少ないチームになった回数を格納している配列
    """
    # print(team_ravel)
    output = deepcopy(womenLess_list)
    min_woman = 100
    for row in gender_count_list:
        if min_woman > row["女"]:
            min_woman = row["女"]
    for i in range(len(team_ravel)):
        if gender_count_list[team_ravel[i]]["女"] == min_woman:
            output[i] += 1
    return output

# 最適化用の関数を定義する
# チームnにおける重複度
def calc_team_cost(n, gender_count_by_team, team_to_member, sameTeam_list, womenLess_list):
    """
    最適化のためのEnergyを計算する関数
    優先したい条件に応じて、重みをつけることで要望に応じた最適化ができる。

    Args:
        n : int
            チームの番号
        gender_count_by_team : dictionary
            チームnの性別の数を格納した辞書型配列
        team_to_member: numpy.ndarray
            チームnに誰が所属しているかを格納する配列
        sameTeam_list : numpy.ndarray
            同じチームになった人を格納する二次元配列
        womenLess_list : numpy.array
            女性が少ないチームになった数を格納する配列
        
    Return:
        g + s + w : int
            計算した重複度（energy）
    """
    team = [int(mem) for mem in team_to_member[n]]
    s = 0
    w = 0

    # 性別の重複度：男女の人数差の二乗
    g = abs(gender_count_by_team[n]['男'] - gender_count_by_team[n]['女']) ** 2

    # チーム構成の重複度：過去にチームメンバー同士が同じチームになった回数*4
    for i in range(len(team)):
        m1 = team[i]
        for j in range(i+1,len(team)):
            m2 = team[j]
            s += sameTeam_list[m1][m2] * 10
    
    # ダミー人間の重複度：ダミー人数の10乗(1つのグループにダミーが二つ以上来るのを避けたいため)
    # d = gender_count_by_team[n]['なし'] * 15
    
    # 女性少数チームの重複度：過去に少数チームになった回数*(5 - 現在の女性人数)
    now_woman = gender_count_by_team[n]['女']
    for i in team:
        w += womenLess_list[i] * (5-now_woman)
    
    return g + s + w

# チーム作成用のクラスを定義する
class GroupingProblem(Annealer):
    """
    最適化を用いてチームを作成するクラス
    """
    def __init__(self, init_state, member_list, sameTeam_list, womenLess_list, totalMember_num, team_num):
        """
        Args:
            init_state : list(ある人が所属しているチームを示す配列, チームnの性別を数えた辞書型配列, チームnに所属する人を格納した二次元配列)
                初期状態(これを毎回同じにして最適化すると、同じグループに続けて所属する人が増えるので毎回適当に変えたほうがいいかも)

            member_list : list(string)
                二次会出席者リスト
            sameTeam_list : numpy.ndarray
                同じチームになった人を格納する二次元配列
            womenLess_list : numpy.array
                女性が少ないチームになった数を格納する配列
            totalMember_num : int
                二次会（会場）にいる人数
            team_num : int
                分けたいチーム数
        """
        super(GroupingProblem, self).__init__(init_state)  # superを使ったクラスの継承 (https://docs.python.org/ja/3/library/functions.html?highlight=super#super)
        self.member_list = member_list
        self.sameTeam_list = sameTeam_list
        self.womenLess_list = womenLess_list
        self.totalMember_num = totalMember_num
        self.team_num = team_num

    def move(self):
        """
        メンバー交換による重複度の差分を返す.

        Return:
            total_cost : int
            a, bの交換前後でのコストの差分
        """
        # ランダムにa,bの２人選ぶ
        a = choice(range(self.totalMember_num))
        b = choice(range(self.totalMember_num))
        # 同一人物だった場合、何もせず終了(重複度の差分は0)
        if a == b:
            return 0
        # a,bそれぞれのチーム
        a_team = self.state[0][a]
        b_team = self.state[0][b]
        # print(a, b)
        # print(a_team, b_team)
        # ２人が同一チームだった場合、何もせず終了(重複度の差分は0)
        if a_team == b_team:
            return 0
         
        # 各チームのメンバー交換前の重複度
        cost_a_before = calc_team_cost(a_team, self.state[1], self.state[2], self.sameTeam_list, self.womenLess_list)
        cost_b_before = calc_team_cost(b_team, self.state[1], self.state[2], self.sameTeam_list, self.womenLess_list)
 
        # aのチームのaの性別の人数
        self.state[1][a_team][self.member_list[a][1]] -= 1
        # print(self.member_list[a][1])
        # bのチームのbの性別の人数
        self.state[1][b_team][self.member_list[b][1]] -= 1
         
        # print(a, self.state[2][a_team])
        # aのチームのリストからaを除く(効率悪いが横着)
        self.state[2][a_team].remove(a)

        # print(b, self.state[2][b_team])
        # bのチームのリストからbを除く(効率悪いが横着)
        self.state[2][b_team].remove(b)

        # a,bの所属チームを交換
        self.state[0][a], self.state[0][b] = self.state[0][b], self.state[0][a]
 
        # aの新しいチームのaの性別の人数
        self.state[1][b_team][self.member_list[a][2]] += 1
        # bの新しいチームのbの性別の人数
        self.state[1][a_team][self.member_list[b][2]] += 1
         
        # aの新しいチームのリストにaを追加
        self.state[2][b_team].append(a)
        # bの新しいチームのリストにbを追加
        self.state[2][a_team].append(b)
        
        # 各チームのメンバー交換後の重複度
        cost_a_after = calc_team_cost(a_team, self.state[1], self.state[2], self.sameTeam_list, self.womenLess_list)
        cost_b_after = calc_team_cost(b_team, self.state[1], self.state[2], self.sameTeam_list, self.womenLess_list)
        total_cost = cost_a_after - cost_a_before + cost_b_after - cost_b_before
        
        return total_cost
              
    # 目的関数
    def energy(self):
        """
        各チームの重複度の和を返す
        
        
        Return:
        各チームの重複度(Energy)の総和
        """

        return sum(calc_team_cost(i, self.state[1], self.state[2], self.sameTeam_list, self.womenLess_list) for i in range(self.team_num))