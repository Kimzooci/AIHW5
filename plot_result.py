import matplotlib.pyplot as plt
import numpy as np
import utils




def read_reward(reward_list, f):
   while True:
      line = f.readline()
      if not line: break
      reward_list.append(float(line))

def get_dr_list(reward_list, mag):
   t = 0
   dr_list = [reward_list[0]]
   for reward in reward_list:
      if t > 100000:
          break
      dr_list.append(0.9*dr_list[-1] + 0.1*reward*mag)
      t += 1
   return dr_list


def meanPlot(listlist):
    listlist = np.array(listlist)
    returnList = []
    smallestLen = len(listlist[0])
    for rlist in listlist:
        if len(rlist) < smallestLen:
            smallestLen = len(rlist)

    for i in range(smallestLen):
        if i > 2000:
            break
        avg = 0
        for rewardlist in listlist:
            avg += rewardlist[i]
        avg /= len(listlist)
        returnList.append(avg)
    return returnList


def plot_result():
    args = utils.get_config()
    save_dir = "Result/{}".format(args.RESULT_SAVE)

    test_object = utils.get_test_object(args)

    files = []

    # 비교 대상 파일 불러오기
    for index1 in range(3):
        data = test_object[index1]
        for index2 in range(3):
            files.append(open("{0}/{1}_{2}_{3}.txt".format(save_dir, data, args.RESULT_SAVE, index2 + 1), 'r'))

    _list = [[] for _ in range(len(files))]

    for l, f in zip(_list, files):
        read_reward(l, f)

    # 그래프 생성 (각 항목마다 3개의 누적 reward의 평균을 이용)

    for i in range(3):
        result = [get_dr_list(_list[i], 1) for i in range(3 * i, 3 * (i + 1))]
        plt.plot(meanPlot(result), label='{0} {1}'.format(test_object[i], args.RESULT_SAVE))

    plt.xlabel('Episode')
    plt.ylabel('Accumulated Reward')
    # 비교 대상에 따라 그래프 제목 변경
    plt.title(args.RESULT_SAVE)
    plt.legend()
    plt.show()

    for f in files:
        f.close()

