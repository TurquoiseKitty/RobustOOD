import gurobipy as gp
from gurobipy import GRB
import numpy as np
import ipdb
from tqdm import tqdm
from joblib import Parallel, delayed


def solve_f_LP(c_test_pred,A,b,task_name):
    num_test = c_test_pred.shape[0]

    if task_name == "knapsack":
        c_test_pred = -c_test_pred

    x_sols = []
    objs = []

    #ipdb.set_trace()

    #x_sols,objs = zip(*Parallel(n_jobs=20)(delayed(solve_f_LP_single)(c_test_pred[i,:],A,b,task_name) for i in range(num_test)))

    for i in tqdm(range(num_test)):
        x_sol,obj = solve_f_LP_single(c_test_pred[i,:],A,b,task_name)
        x_sols.append(x_sol)
        objs.append(obj)


    x_sols = np.array(x_sols)
    objs = np.array(objs)
    if task_name == "knapsack":
        objs = -objs
    return x_sols,objs

def solve_f_LP_single(c,A,b,task_name):
    model = gp.Model("f_LP")
    dim_c = len(c)
    # create the variables
    x = model.addVars(range(dim_c),lb=0,ub=1.0,vtype=GRB.CONTINUOUS,name="x")
    # set the objective
    model.setObjective(gp.quicksum(x[i]*c[i] for i in range(dim_c)),GRB.MINIMIZE)
    # add the constraints
    if task_name=="shortest_path":
        model.addConstrs(gp.quicksum(A[i,j]*x[j] for j in range(dim_c)) == b[i] for i in range(A.shape[0]))
    elif task_name=="knapsack":
        model.addConstrs(gp.quicksum(A[i,j]*x[j] for j in range(dim_c)) <= b[i] for i in range(A.shape[0]))
    
    # set output flag
    model.setParam('OutputFlag', 0)
    # optimize
    model.optimize()
    # get the solution
    x_sol = np.array([x[i].x for i in range(dim_c)])
    obj = model.objVal

    return x_sol,obj



def solve_DRCME(c_train,rho_DX,Imat,I1mat,I2mat,A,b,task_name):
    num_test = Imat.shape[0]

    x_sols = []
    objs = []
    for i in range(num_test):
        x_sol,obj = solve_DRCME_single(c_train,rho_DX[i,:],Imat[i,:],I1mat[i,:],I2mat[i,:],A,b,task_name)
        x_sols.append(x_sol)
        objs.append(obj)

    x_sols = np.array(x_sols)
    objs = np.array(objs)
    if task_name == "knapsack":
        objs = -objs
    return x_sols,objs

def solve_DRCME_single(c_train,rho_DX,Imat,I1mat,I2mat,A,b,task_name):
    Ilist = np.where(Imat==1)[0]

    I2list = np.where(I2mat==1)[0]

    I3list = np.where(rho_DX>=0)[0]

    dim_c = c_train.shape[1]
    #create the model
    m = gp.Model("DRCME")

    # create the variables
    x = m.addVars(range(dim_c),lb=0,ub=1.0,vtype=GRB.CONTINUOUS,name="x")
    lamb = m.addVar(vtype=GRB.CONTINUOUS,name="lamb")
    u = m.addVars(Ilist,lb=0.0,vtype=GRB.CONTINUOUS,name="u")
    t = m.addVars(I3list,lb=0.0,vtype=GRB.CONTINUOUS,name="t")

    # set objective
    m.setObjective(lamb,GRB.MINIMIZE)

    # add constraints
    m.addConstrs(u[i]>=0 for i in I2list)
    if task_name == "shortest_path":
        m.addConstrs(lamb+u[i]>= gp.quicksum(x[j]*c_train[i,j] for j in range(dim_c))+rho_DX[i]*t[i] for i in I3list)
    elif task_name == "knapsack":
        m.addConstrs(lamb+u[i]>= gp.quicksum(-x[j]*c_train[i,j] for j in range(dim_c))+rho_DX[i]*t[i] for i in I3list)
    m.addConstr(gp.quicksum(u)<=0)
    m.addConstrs(t[i]>=x[j] for i in I3list for j in range(dim_c))
    m.addConstrs(t[i]>=-x[j] for i in I3list for j in range(dim_c))

    # add task constraints
    if task_name=="shortest_path":
        m.addConstrs(gp.quicksum(A[i,j]*x[j] for j in range(dim_c))==b[i] for i in range(A.shape[0]))
    elif task_name=="knapsack":
        m.addConstrs(gp.quicksum(A[i,j]*x[j] for j in range(dim_c))<=b[i] for i in range(A.shape[0]))

    # set output flag = 0
    m.setParam('OutputFlag', 0)

    #optimize
    m.optimize()

    if m.status == GRB.Status.OPTIMAL:
        x_sol = np.array([x[i].x for i in range(dim_c)])
        obj = m.objVal
    else:
        raise Exception("did not find optimal solution")
    

    return x_sol,obj

def solve_wasserstein1_LP(eps,A,b,T,phi,w,f,r,task_name):
    if T is None:
        T = np.zeros((1,A.shape[1]))
        phi = np.zeros(1)

    num_test = f.shape[0]
    x_sols = []
    objs = []

    if task_name == "knapsack":
        f = -f
        r = -r

    x_sols,objs = zip(*Parallel(n_jobs=20)(delayed(solve_wasserstein1_LP_single)(eps,A,b,T,phi,w[i,:],f[i,:],r,task_name) for i in range(num_test)))

    x_sols = np.array(x_sols)
    objs = np.array(objs)

    if task_name == "knapsack":
        objs = -objs
    return x_sols,objs

def solve_wasserstein1_LP_single(eps,A,b,T,phi,w,f,r,task_name):
    Tm = T.shape[0]

    x_dim = A.shape[1]

    Am = A.shape[0]
    N = len(w)
    #create the model
    m = gp.Model("wasserstein")

    #create the variables
    x = m.addVars(range(x_dim),lb=0,ub=1.0,vtype=GRB.CONTINUOUS,name="x")
    s = m.addVars(range(N),lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name="s")
    pi_inds = [(i,j) for i in range(N) for j in range(Tm)]
    pi = m.addVars(pi_inds,lb=0.0,vtype=GRB.CONTINUOUS,name="pi")
    lamb = m.addVar(lb=0.0,vtype=GRB.CONTINUOUS,name="lamb")

    #set objective
    m.setObjective(gp.quicksum(w[i]*s[i] for i in range(N))+eps*lamb,GRB.MINIMIZE)

    #add constraints
    # add the constraints for distributionally robust
    for i in range(N):
        Txi = T@(r[i,:].T)
        #constraint 1
        left = gp.quicksum(phi[j]*pi[i,j] for j in range(Tm))
        left += gp.quicksum(f[j]*x[j] for j in range(x_dim))
        left += gp.quicksum(x[j]*r[i,j] for j in range(x_dim))
        left -= gp.quicksum(pi[i,j]*Txi[j] for j in range(Tm))
        m.addConstr(left<=s[i])

        #constraint 2
        m.addConstrs(x[j]-gp.quicksum(T[jj,j]*pi[i,jj] for jj in range(Tm))<=lamb for j in range(x_dim))
        m.addConstrs(x[j]-gp.quicksum(T[jj,j]*pi[i,jj] for jj in range(Tm))>=-lamb for j in range(x_dim))

    # add task constraints
    if task_name=="shortest_path":
        m.addConstrs(gp.quicksum(A[i,j]*x[j] for j in range(x_dim))==b[i] for i in range(Am))
    elif task_name=="knapsack":
        m.addConstrs(gp.quicksum(A[i,j]*x[j] for j in range(x_dim))<=b[i] for i in range(Am))

    #optimize
    # not output log
    m.setParam('OutputFlag', False)
    m.optimize()

    #get the solution and objectives
    if m.status==2:
        x_sol = np.array([x[j].x for j in range(x_dim)])
        s_sol = np.array([s[i].x for i in range(N)])
        pi_sol = np.array([pi[i,j].x for i in range(N) for j in range(Tm)])
        pi_sol = pi_sol.reshape(N,Tm)
        lamb_sol = lamb.x
        obj = m.objVal
    else:
        ipdb.set_trace()
    
    return x_sol,obj


def solve_wasserstein1(eps,A,b,T,phi,w,a,f,r,gamma,o,task_name):
    '''
    solve the distributionally robust problem with 1-Wasserstein distance.
    If this is a maximization problem, remenber to multiply -1 to the objective coef.
    '''
    K = len(a)
    x_dim = T.shape[1]
    Tm = T.shape[0]
    Am = A.shape[0]
    N = len(w)
    v_dim = o.shape[1]
    #create the model
    m = gp.Model("wasserstein")

    #create the variables
    x = m.addMVars(range(x_dim),lb=0,ub=1.0,vtype=GRB.CONTINUOUS,name="x")
    v = m.addMVars(range(v_dim),vtype=GRB.CONTINUOUS,name="v")
    s = m.addMVars(range(N),lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name="s")
    pi_inds = [(i,k,j) for i in range(N) for k in range(K) for j in range(Tm)]
    pi = m.addMVars(pi_inds,lb=0.0,vtype=GRB.CONTINUOUS,name="pi")
    lamb = m.addVar(lb=0.0,vtype=GRB.CONTINUOUS,name="lamb")

    #set objective
    m.setObjective(gp.quicksum(w[i]*s[i] for i in range(N))+eps*lamb,GRB.MINIMIZE)

    #add constraints
    # add the constraints for distributionally robust
    for i in range(N):
        Txi = T@(r[i,:].T)
        for k in range(K):
            #constraint 1
            left = gp.quicksum(phi[j]*pi[i,k,j] for j in range(Tm))
            left += a[k]*gp.quicksum(f[j]*x[j] for j in range(x_dim))
            left += gamma[k]
            left += gp.quicksum(o[k,j]*v[j] for j in range(v_dim))
            left += a[k]*gp.quicksum(x[j]*r[i,j] for j in range(x_dim))
            left -= gp.quicksum(pi[i,k,j]*Txi[j] for j in range(Tm))
            m.addConstr(left<=s[i])

            #constraint 2
            m.addConstrs(a[k]*x[j]-gp.quicksum(T[jj,j]*pi[i,k,jj] for jj in range(Tm))<=lamb for j in range(x_dim))
            m.addConstrs(a[k]*x[j]-gp.quicksum(T[jj,j]*pi[i,k,jj] for jj in range(Tm))>=-lamb for j in range(x_dim))

    # add task constraints
    if task_name=="shortest_path":
        m.addConstrs(gp.quicksum(A[i,j]*x[j] for j in range(x_dim))==b[i] for i in range(Am))
    elif task_name=="knapsack":
        m.addConstrs(gp.quicksum(A[i,j]*x[j] for j in range(x_dim))<=b[i] for i in range(Am))

    #optimize
    # not output log
    m.setParam('OutputFlag', False)
    m.optimize()

    #get the solution and objectives
    if m.status==2:
        x_sol = np.array([x[j].x for j in range(x_dim)])
        v_sol = np.array([v[j].x for j in range(v_dim)])
        s_sol = np.array([s[i].x for i in range(N)])
        pi_sol = np.array([pi[i,k,j].x for i in range(N) for k in range(K) for j in range(Tm)])
        pi_sol = pi_sol.reshape(N,K,Tm)
        lamb_sol = lamb.x
        obj = m.objVal
    else:
        ipdb.set_trace()
    
    return x_sol,obj

def solve_wasserstein_res(eps,A,b,T,phi,f,q,res_train,task_name):
    ''' solve the residual-based DRO problem'''
    num_test_samples = f.shape[0]
    num_train_samples = res_train.shape[0]
    dim_c = f.shape[1]
    Am = A.shape[0]
    if T is not None:
        Tm = T.shape[0]
    else:
        Tm = 1
        T = np.zeros((1,dim_c))
        phi = np.zeros(1)

    x_sols = []
    objs = []

    if task_name == "knapsack":
        f = -f
        q = -q


    x_sols, objs = zip(*Parallel(n_jobs=20)(delayed(solve_wasserstein_res_single)(eps,A,b,T,phi,f[test_ind,:],q[test_ind,:],res_train,task_name) for test_ind in range(num_test_samples)))
    """
    for test_ind in tqdm(range(num_test_samples)):
        x_sol,obj = solve_wasserstein_res_single(eps,A,b,T,phi,f[test_ind,:],q[test_ind,:],res_train,task_name)
        x_sols.append(x_sol)
        objs.append(obj)
    """

    x_sols = np.array(x_sols)
    objs = np.array(objs)

    return x_sols,objs

def solve_wasserstein_res_single(eps,A,b,T,phi,f,q,res_train,task_name):
    num_train_samples = res_train.shape[0]
    dim_c = A.shape[1]
    Am = A.shape[0]
    if T is not None:
        Tm = T.shape[0]
    else:
        Tm = 1
        T = np.zeros((1,dim_c))
        phi = np.zeros(1)

    # repeat q to match the shape of res_train
    q_res = q*res_train

    model = gp.Model("wasserstein_res")
    #create the variables
    if task_name == "shortest_path":
        x = model.addVars(range(dim_c),lb=0,ub=1.0,vtype=GRB.BINARY,name="x")
        
    elif task_name == "knapsack":
        x = model.addVars(range(dim_c),lb=0,ub=1.0,vtype=GRB.CONTINUOUS,name="x")

    lamb = model.addVar(lb=0.0,vtype=GRB.CONTINUOUS,name="lamb")
    s = model.addVars(range(num_train_samples),lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name="s")
    pi_ind = [(i,j) for i in range(num_train_samples) for j in range(Tm)]
    pi = model.addVars(pi_ind,lb=0.0,vtype=GRB.CONTINUOUS,name="pi")

    #set objective
    model.setObjective(gp.quicksum(s[i] for i in range(num_train_samples))/num_train_samples+eps*lamb,GRB.MINIMIZE)

    # set constraints
    # add the constraints for distributionally robust
    for i in range(num_train_samples):
        T_res = T@(res_train[i,:].T)
        left = gp.quicksum(phi[j]*pi[i,j] for j in range(Tm))
        left += gp.quicksum(f[j]*x[j] for j in range(dim_c))
        left += gp.quicksum(q_res[i,j]*x[j] for j in range(dim_c))

            
        left -= gp.quicksum(pi[i,j]*T_res[j] for j in range(Tm))
        model.addConstr(left<=s[i])

        for j in range(dim_c):
            model.addConstr(q[j]*x[j]-gp.quicksum(T[jj,j]*pi[i,jj] for jj in range(Tm))<=lamb)
            model.addConstr(q[j]*x[j]-gp.quicksum(T[jj,j]*pi[i,jj] for jj in range(Tm))>=-lamb)
    # add task constraints
    if task_name=="shortest_path":
        model.addConstrs(gp.quicksum(A[i,j]*x[j] for j in range(dim_c))==b[i] for i in range(Am))
    elif task_name=="knapsack":
        model.addConstrs(gp.quicksum(A[i,j]*x[j] for j in range(dim_c))<=b[i] for i in range(Am))
    
    #optimize
    # not output log
    model.setParam('OutputFlag', False)
    model.optimize()

    
    if model.status==2:
        x_sol = np.array([x[j].x for j in range(dim_c)])
        s_sol = np.array([s[i].x for i in range(num_train_samples)])
        pi_sol = np.array([pi[i,j].x for i in range(num_train_samples) for j in range(Tm)])
        pi_sol = pi_sol.reshape(num_train_samples,Tm)
        lamb_sol = lamb.x
        obj = model.objVal
    else:
        ipdb.set_trace()

    if task_name == "knapsack":
        obj = -obj

    return x_sol,obj


def get_spp_Ab():
    '''
    Construct the constraint matrix A and vec b for the shortest path problem starting from [0,0] to [4,4] on a 5*5 grid network.
    '''
    n = 40
    m = 25
    A = np.zeros((m,n))
    b = np.zeros(m)
    for i in range(5):
        for j in range(5):
            v_idx = i*5+j
            if j!=4:
                #edge that point from v_idx to its right neighbor
                edge_to_right_idx = 9*i+j
                A[v_idx,edge_to_right_idx] = 1
            if i!=4:
                #edge that point from v_idx to its bottom neighbor
                edge_to_bottom_idx = 9*i+4+j
                A[v_idx,edge_to_bottom_idx] = 1
            if j!=0:
                #edge that point from the left neighbor to v_idx
                edge_from_left_idx = 9*i+j-1
                A[v_idx,edge_from_left_idx] = -1
            if i!=0:
                #edge that point from the top neighbor to v_idx
                edge_from_top_idx = 9*(i-1)+4+j
                A[v_idx,edge_from_top_idx] = -1
            if i==0 and j==0:
                b[v_idx] = 1
            elif i==4 and j==4:
                b[v_idx] = -1
    return A,b

def get_kp_Ab(price,budget):
    '''
    Construct the constraint matrix A and vec b for the 0-1 knapsack problem with 10 items and 10 knapsacks.
    '''
    n = 10
    m = 1
    A = np.ones((m,n))
    b = np.zeros(m)
    for j in range(n):
        A[0,j] = price[j]
    b[0] = budget
    return A,b



