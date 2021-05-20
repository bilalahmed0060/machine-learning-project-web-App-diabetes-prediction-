import pandas as pd
import numpy




mturk_res=pd.read_csv(r"C:\Users\CCS LAPTOP HYD\Downloads\hw7_data.csv")
num_worker = mturk_res["WorkerId"]
# print(len(num_worker))
num_worker = num_worker.unique()
rows = mturk_res.shape
row = rows[0]


num_of_HITS = []
n_q_c = []
p_q_c = []
inc_neg = []
inc_pos = []
c_neg_workers = []
c_neg_and_pass = []
hid = []
neg_qual = mturk_res['Answer.neg_qual_ctrl'].unique()
qual = mturk_res['Answer.pos_qual_ctrl_3'].unique()

# print(neg_qual[-1])
for i in range(len(num_worker)):
    num = 0
    n_q = 0
    p_c = 0
    for j in range(row):
        if num_worker[i] == mturk_res['WorkerId'][j]:
            num = num+1
    num_of_HITS.append(num)
for i in range(len(num_worker)):
    if num_of_HITS[i] < 5:
       num_worker1 =  numpy.delete(num_worker, (i), axis=0)



print("====num_of_worker=====")
# print(num_worker.shape)
for i in range(row):
    if mturk_res['Answer.neg_qual_ctrl'][i] == neg_qual[-1]:
        c_neg_workers.append(mturk_res['WorkerId'][i])
# print("==correct negative===")
print(len(c_neg_workers))

# print(d_w)
# print(n_w)
# for i in range(len(worker_id)):
#     index = 0
#     for j in range(row):
#         if d_w[i] == mturk_res['WorkerId'][j]:
#             index = mturk_res[mturk_res['WorkerId'][j] == d_w[i]]
#         if j==rows[0]-1 and index != 0:
#             mturk_res.drop(index, inplace=True)
#             index = 0




for j in range(row):
    if mturk_res['Answer.neg_qual_ctrl'][j] == neg_qual[-1]:
        n = 0
        if mturk_res['Answer.pos_qual_ctrl_1'][j] == "Yes" or mturk_res['Answer.pos_qual_ctrl_1'][j] == "No" or  mturk_res['Answer.pos_qual_ctrl_1'][j] == "Naa" or  mturk_res['Answer.pos_qual_ctrl_1'][j] == "nan":
            if mturk_res['Answer.pos_qual_ctrl_2'][j] == 'Yes' and  mturk_res['Answer.pos_qual_ctrl_3'][j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_4'][j]=='Yes' and mturk_res['Answer.pos_qual_ctrl_5'][j] == 'Yes':

                c_neg_and_pass.append(mturk_res["WorkerId"][j])
                n = 1

        elif n==0 and mturk_res['Answer.pos_qual_ctrl_2'][j] == "Yes" or mturk_res['Answer.pos_qual_ctrl_2'][j] == "No" or mturk_res['Answer.pos_qual_ctrl_2'][j] == "Naa" or  mturk_res['Answer.pos_qual_ctrl_2'][j] == "nan":
                if mturk_res['Answer.pos_qual_ctrl_1'][j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_3'][j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_4'][j]=='Yes' and mturk_res['Answer.pos_qual_ctrl_5'][j] == 'Yes':
                    c_neg_and_pass.append(mturk_res["WorkerId"][j])
                    n = 1


        elif n==0 and mturk_res['Answer.pos_qual_ctrl_3'][j] == "Yes" or mturk_res['Answer.pos_qual_ctrl_3'][j] == "No" or mturk_res['Answer.pos_qual_ctrl_3'][j] == "Naa" or  mturk_res['Answer.pos_qual_ctrl_3'][j] == "nan":
                if mturk_res['Answer.pos_qual_ctrl_1'][j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_2'][j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_4'][j]=='Yes' and mturk_res['Answer.pos_qual_ctrl_5'][j] == 'Yes':
                    c_neg_and_pass.append(mturk_res["WorkerId"][j])
                    n = 1


        elif n==0 and mturk_res['Answer.pos_qual_ctrl_4'][j] == "Yes" or mturk_res['Answer.pos_qual_ctrl_4'][j] == "No" or mturk_res['Answer.pos_qual_ctrl_4'][j] == "Naa" or  mturk_res['Answer.pos_qual_ctrl_4'][j] == "nan":
                if mturk_res['Answer.pos_qual_ctrl_1'][j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_2'][j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_3'][j]=='Yes' and mturk_res['Answer.pos_qual_ctrl_5'][j] == 'Yes':
                    c_neg_and_pass.append(mturk_res["WorkerId"][j])
                    n = 1

        elif n==0 and mturk_res['Answer.pos_qual_ctrl_5'][j] == "Yes" or mturk_res['Answer.pos_qual_ctrl_5'][j] == "No" or mturk_res['Answer.pos_qual_ctrl_5'][j] == "Naa" or  mturk_res['Answer.pos_qual_ctrl_5'][j] == "nan":
                if mturk_res['Answer.pos_qual_ctrl_1'][j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_2'][j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_3'][j]=='Yes' and mturk_res['Answer.pos_qual_ctrl_4'][j] == 'Yes':
                    c_neg_and_pass.append(mturk_res["WorkerId"][j])
                    n = 1

print(len(c_neg_and_pass))
c_neg = []
# for i in range(row):
#     if mturk_res['Answer.neg_qual_ctrl'][i] != neg_qual[-1]:
#         mturk_res = mturk_res.loc[mturk_res['HITId']]
#         c_neg.append(mturk_res['Answer.neg_qual_ctrl'][i])


wor = set(c_neg_and_pass)
wor = list(wor)
c_no_worker = []
c_worker_id = []
print("wor ",len(wor))
for i in range(len(wor)):
    count = 0
    for j in range(len(c_neg_and_pass)):
        if wor[i] == c_neg_and_pass[j]:
            count = count + 1
        if j == len(c_neg_and_pass)-1:
            c_worker_id.append(wor[i])
            c_no_worker.append(count)




print("c_no_worker ",c_no_worker)
print("c_worker_id ",c_worker_id)
c_total= []
totall = []

no_worker = []
worker_id = []
d_w = []
n_w = []
print(row)
for i in range(len(num_worker)):
    count = 0
    for j in range(row):
        if mturk_res['WorkerId'][j]==num_worker[i]:
            count = count+1
        if j==rows[0]-1:
            worker_id.append(num_worker[i])
            no_worker.append(count)
d = {'worker_id': worker_id,
     'hist':no_worker}
d = pd.DataFrame(d)
# d.to_csv('compl.csv')
percentage = []
new_worker_id = []
worker = numpy.array(wor)
for i in range(len(wor)):
    n = 0
    for j in range(len(c_neg_and_pass)):
        if wor[i] == c_neg_and_pass[j]:
            n = n+1
    c_total.append(n)

for i in range(len(wor)):
    add = 0
    for j in range(len(worker_id)):
        if wor[i] == worker_id[j]:
           add = add+1

    totall.append(add)
totall = numpy.array(totall)


c_total = numpy.array(c_total)

percentage = []
ind = []
for i in range(len(c_total)):
    if totall[i]>=5:
        per = (c_total[i]*100)/totall[i]
        per = float("{0:.3f}".format(per))
        percentage.append(per)
    else:
        ind.append(i)
worker = numpy.delete(worker, ind)
c_total = numpy.delete(c_total,ind)
totall = numpy.delete(totall,ind)

mturk_res = mturk_res.loc[mturk_res['Answer.neg_qual_ctrl'] == 'Yes']
print("mturk",mturk_res.shape)
mturk_res = pd.DataFrame(mturk_res)
c_neg_and_pass = []
print(mturk_res)
print(mturk_res['Answer.neg_qual_ctrl'].values[0])
index = []
for j in range(306):
    n = 0
    if mturk_res['Answer.pos_qual_ctrl_2'].values[j] == 'Yes' and  mturk_res['Answer.pos_qual_ctrl_3'].values[j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_4'].values[j]=='Yes' and mturk_res['Answer.pos_qual_ctrl_5'].values[j] == 'Yes':
        # c_neg_and_pass.append(mturk_res["WorkerId"][j])
        n = 1

    elif mturk_res['Answer.pos_qual_ctrl_1'].values[j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_3'].values[j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_4'].values[j]=='Yes' and mturk_res['Answer.pos_qual_ctrl_5'].values[j] == 'Yes':
        # c_neg_and_pass.append(mturk_res["WorkerId"][j])
        n = 1


    elif mturk_res['Answer.pos_qual_ctrl_1'].values[j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_2'].values[j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_4'].values[j]=='Yes' and mturk_res['Answer.pos_qual_ctrl_5'].values[j] == 'Yes':
        # c_neg_and_pass.append(mturk_res["WorkerId"][j])
        n = 1


    elif mturk_res['Answer.pos_qual_ctrl_1'].values[j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_2'].values[j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_3'].values[j]=='Yes' and mturk_res['Answer.pos_qual_ctrl_5'].values[j] == 'Yes':
        # c_neg_and_pass.append(mturk_res["WorkerId"][j])
        n = 1

    elif mturk_res['Answer.pos_qual_ctrl_1'].values[j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_2'].values[j] == 'Yes' and mturk_res['Answer.pos_qual_ctrl_3'].values[j]=='Yes' and mturk_res['Answer.pos_qual_ctrl_4'].values[j] == 'Yes':
        # c_neg_and_pass.append(mturk_res["WorkerId"][j])
        n = 1
    else:
        index.append(mturk_res.index[j])
        #
mturk_res = mturk_res.drop(labels=index,axis=0)
data = {"worker_id":mturk_res['WorkerId'].values,
        "Answer.neg_qual_ctrl":mturk_res['Answer.neg_qual_ctrl'].values,
        "Answer.pos_qual_ctrl_1":mturk_res['Answer.pos_qual_ctrl_1'].values,
        "Answer.pos_qual_ctrl_2":mturk_res['Answer.pos_qual_ctrl_2'].values,
        "Answer.pos_qual_ctrl_3":mturk_res['Answer.pos_qual_ctrl_3'].values,
        "Answer.pos_qual_ctrl_4":mturk_res['Answer.pos_qual_ctrl_4'].values,
        "Answer.pos_qual_ctrl_5":mturk_res['Answer.pos_qual_ctrl_5'].values,
        }
data = pd.DataFrame(data)
# data.to_csv("hss.csv")
print(data.shape)
worker_hist = []
worker_name = []
for i in range(191):
    n = 0
    for j in range(191):
        if data['worker_id'][i] == data['worker_id'][j]:
            n = n+1
    if n >= 5:
        worker_hist.append(n)
        worker_name.append(data['worker_id'][i])
print(worker_hist)
print(worker_name)
data1 = {"worker_id":worker_name,
         "num_of_hist":worker_hist

}
data1 = pd.DataFrame(data1)
# data1.to_csv("greater 5_hst.csv")

dl = d.shape[0]
data1l = data1.shape[0]

sel_worker = []
percentage = []
for i in range(dl):
    n = 0
    for j in range(data1l):
        if d['worker_id'][i] == data1['worker_id'][j]:
            if d['worker_id'][i] not in sel_worker:
                per = (data1['num_of_hist'][j]*100)/d['hist'][i]
                sel_worker.append(d['worker_id'][i])
                per = float("{0:.3f}".format(per))
                percentage.append(per)

print(sel_worker)
print(percentage)
# # intialise data of lists.





