import csv

a = {"质押人:发起质押的个人", "质权人:接受质押的个人", "股份股权转让人:发起股份股权转让的个人",
     "受转让人:接受股份股权转让的个人", "原告(个人)", "被告(个人)", "减持方(通常为人，少数情况为组织)", "被减持方",
     "收购方:发起收购的个人", "担保人:发起担保的个人", "发起合同签署的自然人(甲方)", "接受合同签署的自然人(乙方)"}
b = {"质押公司:发起质押的公司", "质权公司:接受质押的公司", "股份股权转让公司:发起股份股权转让的公司",
     "受转让公司:接受股份股权转让的公司", "标的公司:指收购行为中的收购对象公司", "原告(公司)", "被告(公司)",
     "发起投资的组织或单位(投资方)", "被投资的组织或单位(被投资方)", "收购公司:发起收购的公司",
     "被收购公司:接受收购的公司", "担保公司:发起担保的公司", "受担保公司:接受担保的公司", "中标方(公司)",
     "招标方(单位或组织)", "发起合同签署的组织或单位(甲方)", "接受合同签署的组织或单位(乙方)", "裁判单位(法院)"}
c = {"质押金额", "质押数量", "转让交易金额", "转让数量", "投资金额", "收购金额", "收购股份数量", "担保金额", "中标金额",
     "成交额", "判决金额"}
d = {"质押比例", "转让比例", "减持的股份占个人股份百分比", "减持的股份占公司股份的百分比", "收购股份比例"}
e = {"质押日期", "转让日期", "起诉日期", "日期", "收购日期", "担保日期", "中标日期", "判决日期"}


def main(fewness):
    pre = [0, 0, 0, 0, 0]
    golden = [0, 0, 0, 0, 0]
    count = [0, 0, 0, 0, 0]
    csv_file = f"./output/HMEAE_transfer_model_eval_few_{fewness}.csv"
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if 0 < int(row[4]) < int(row[5]):
                if row[0].split('"')[1] in a:
                    golden[0] += 1
                if row[0].split('"')[1] in b:
                    golden[1] += 1
                if row[0].split('"')[1] in c:
                    golden[2] += 1
                if row[0].split('"')[1] in d:
                    golden[3] += 1
                if row[0].split('"')[1] in e:
                    golden[4] += 1
            if 0 < int(row[3]):
                if row[0].split('"')[1] in a:
                    count[0] += 1
                    if row[4] <= row[3] <= row[5]:
                        pre[0] += 1
                if row[0].split('"')[1] in b:
                    count[1] += 1
                    if row[4] <= row[3] <= row[5]:
                        pre[1] += 1
                if row[0].split('"')[1] in c:
                    count[2] += 1
                    if row[4] <= row[3] <= row[5]:
                        pre[2] += 1
                if row[0].split('"')[1] in d:
                    count[3] += 1
                    if row[4] <= row[3] <= row[5]:
                        pre[3] += 1
                if row[0].split('"')[1] in e:
                    count[4] += 1
                    if row[4] <= row[3] <= row[5]:
                        pre[4] += 1
    print(f'{fewness}', end=': ')
    for i in range(5):
        prediction = pre[i] / count[i]
        recall = pre[i] / golden[i]
        print(prediction, end=',')
    print()


if __name__ == '__main__':
    for i in [100, 50, 20, 10, 4]:
        main(i)
