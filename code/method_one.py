from pulp import *
import numpy as np


def run(index, work_days, A_times, B_times, C_times, D_times, qualify, s):
    # 员工人数
    count = len(index)
    # 已经安排班次的次数
    drive_times = A_times+B_times+C_times+D_times
    # 工作率：已经安排班次的次数/已经工作的天数
    work_rate = drive_times/(work_days+1)
    # 是否有经验
    experienced = np.array([1 if i>0 else 0 for i in drive_times])

    # 优化问题
    my = LpProblem("Work assigning problem", LpMinimize)
    # 定义各组0-1变量
    # 某员工是否参加A工作
    A = LpVariable.matrix("A",(index,),0,1,LpInteger)
    if count>=8:
        B = LpVariable.matrix("B",(index,),0,1,LpInteger)
        if count >=11:
            C = LpVariable.matrix("C",(index,),0,1,LpInteger)
            if count >=13:
                D = LpVariable.matrix("D",(index,),0,1,LpInteger)

    # 目标函数第一部分，将约束条件4转化为目标函数，每个人的ABCD班次均衡，即优先安排参与班次少的情形
    obj = ''
    for i in range(count):
        tmp = np.array([A_times[i],B_times[i],C_times[i],D_times[i]])
        tmp = tmp - tmp.min()
        obj += lpDot(A[i],tmp[0])
        if count>=8:
            obj += lpDot(B[i],tmp[1])
            if count>=11:
                obj += lpDot(C[i],tmp[2])
                if count>=13:
                    obj += lpDot(D[i],tmp[3])

    '''
    #约束2,3,5
    if count<=7:
        #obj += -lpSum(lpDot(A,qualify))
        obj += -lpSum(lpDot(A,experienced))
        obj += -lpSum(lpDot(A,s))
    elif count<=10:
        #obj += -lpSum(lpDot(A,qualify))-lpSum(lpDot(B,qualify))
        obj += -lpSum(lpDot(A,experienced))-lpSum(lpDot(B,experienced))
        obj += -lpSum(lpDot(A,s))-lpSum(lpDot(B,s))
    '''

    # 两个目标的权重分配
    lambda1 = 1
    # 目标函数第二部分，每个人的工作安排均衡
    if count<=7:
        my += lambda1 * obj + lpDot(A,work_rate)
    elif 8<=count<=10:
        my += lambda1 * obj + lpDot(A,work_rate)+lpDot(B,work_rate)
    elif 11<=count<=12:
        my += lambda1 * obj + lpDot(A,work_rate)+lpDot(B,work_rate)+lpDot(C,work_rate)
    elif 13<=count<=15:
        my += lambda1 * obj + lpDot(A,work_rate)+lpDot(B,work_rate)+lpDot(C,work_rate)+lpDot(D,work_rate)
    
    # 约束条件0，每人每天至多参与一项排班
    for i in range(count):
        if count<=7:
            my += A[i]<=1
        elif 8<=count<=10:
            my += A[i]+B[i]<=1
        elif 11<=count<=12:
            my += A[i]+B[i]+C[i]<=1
        elif 13<=count<=15:
            my += A[i]+B[i]+C[i]+D[i]<=1
    
    # 约束条件1,每个任务都有2人做
    my += lpSum(A)==2
    if count>=8:
        my += lpSum(B)==2
        if count >=11:
            my += lpSum(C)==2
            if count >=13:
                my += lpSum(D)==2

    # 约束条件2，至少一位有驾驶资格的
    my += lpSum(lpDot(A,qualify))>=1
    if count>=8 :
        my += lpSum(lpDot(B,qualify))>=1
        if count >=11:
            my += lpSum(lpDot(C,qualify))>=1
            if count >=13:
                my += lpSum(lpDot(D,qualify))>=1

    # 约束条件3，至少一位有经验的
    my += lpSum(lpDot(A,experienced))>=1
    if count>=8:
        my += lpSum(lpDot(B,experienced))>=1
        if count >=11:
            my += lpSum(lpDot(C,experienced))>=1
            if count >=13:
                my += lpSum(lpDot(D,experienced))>=1

    '''
    #约束条件4，班次均衡，即优先安排参与班次少的情形
    for i in range(count):
        tmp = np.array([A_times[i],B_times[i],C_times[i],D_times[i]])
        tmp = tmp - tmp.min()
        my += lpDot(A[i],tmp[0])==0
        if count>=8:
            my += lpDot(B[i],tmp[1])==0
            if count>=11:
                my += lpDot(C[i],tmp[2])==0
                if count>=13:
                    my += lpDot(D[i],tmp[3])==0
    '''
    
    # 约束条件5，历史事故数多的要有少的来搭配
    my += lpSum(lpDot(A,s))>=0
    if count>=8:
        my += lpSum(lpDot(B,s))>=0
        if count >=11:
            my += lpSum(lpDot(C,s))>=0
            if count >=13:
                my += lpSum(lpDot(D,s))>=0

    """
    为了求解次优解增加约束
    非A=（10，6）；B=（1，8）
    my += A[9]+A[5]+B[0]+B[7]<=3
    my += A[9]+A[5]+B[0]+B[8]<=3
    """
    my.solve()
    
    solution = []
    print("Status:", LpStatus[my.status])
    
    for v in my.variables():
        if v.varValue==1:
            # 由于数组是从0开始，而员工序号是从1开始，因此需要加以修改
            tmp = v.name.split('_')
            tmp[1] = str(int(tmp[1])+1)
            print("员工"+tmp[1]+'负责班次'+tmp[0])
        solution.append(v.varValue)
    
    print("objective=", value(my.objective))
    return solution, my.objective


def renew(times,add,index):
    for (i,idx) in enumerate(index):
        times[idx] += add[i]
    return times

# 员工编号
index = list(range(1,11))

# 已经工作的天数
work_days = np.array([4,6,8,5,8,11,4,5,3,1,-2,-3,-5,-5,-6])
all_days = np.array([6,11,9,10,13,15,6,10,5,10,6,10,4,8,5])

# 每个人安排过A的次数
A_times = np.array([2,1,1,1,2,1,1,1,0,0,0,0,0,0,0])
B_times = np.array([0,1,2,1,1,1,1,0,0,0,0,0,0,0,0])
C_times = np.array([0,0,1,1,0,1,0,1,1,0,0,0,0,0,0])
D_times = np.array([0,0,0,0,1,1,0,0,1,0,0,0,0,0,0])

# 是否有驾驶资格
qualify = np.array([1,1,1,1,1,1,1,0,0,0,1,0,1,0,1])

# 事故率，1代表事故次数少（0,1),0表示事故次数中等(2,3),-1表示事故次数多（>3）
s = np.array([1,1,1,0,0,-1,0,-1,0,1,1,1,1,1,1])

day = 1
while(1):
    print('Day '+str(day))
    # prepare data
    index_raw = np.array(np.where(work_days>=0)[0])
    index = []
    for i in range(len(index_raw)):
        idx = index_raw[i]
        if work_days[idx] < all_days[idx]:
            index.append(idx)
    
    work_days_tmp = work_days[index]
    # 第二问
    all_days_tmp = all_days[index]

    A_times_tmp = A_times[index]
    B_times_tmp = B_times[index]
    C_times_tmp = C_times[index]
    D_times_tmp = D_times[index]
    qualify_tmp = qualify[index]
    s_tmp = s[index]
    count = len(index)

    if count < 5:
        break
    # get solution
    # 第一问
    solution, objective = run(index, work_days_tmp, A_times_tmp, B_times_tmp, 
                             C_times_tmp, D_times_tmp, qualify_tmp, s_tmp)
    work_days+=1
    A_add = np.array(solution[0:count])
    A_times = renew(A_times,A_add,index)
    if count>=8:
        B_add = np.array(solution[count:2*count])
        B_times = renew(B_times,B_add,index)
        if count >=11:
            C_add = np.array(solution[2*count:3*count])
            C_times = renew(C_times,C_add,index)
            if count >=13:
                D_add = np.array(solution[3*count:])
                D_times = renew(D_times,D_add,index)
    day+=1
