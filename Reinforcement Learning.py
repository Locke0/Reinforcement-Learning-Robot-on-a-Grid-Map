import numpy as np
import matplotlib.pyplot as plt

# Set up grid map
gmap = np.ones((10,10))
for i in range(10):
    for j in range(10):
        if i == 0 or j ==0:
            gmap[i, j] = 0
        if i == 9 or j == 9:
            gmap[i, j] = 0    
gmap[4, 4:8] = 0
gmap[5, 7] = 0
gmap[7,4:6] = 0
gmap[3:7, 2] = 0

# States
S = [0]* 100
# np.zeros((100, 1))
for i in range(10):
    for j in range(10):
        S[10*i+j] = 10*i+j


# Actions go north, south, east, west
A = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

N_S = len(S)
N_A = len(A)

def transition_matrix(S, A, gmap, r_m):  
    N_S = len(S)
    N_A = len(A)
    
    # Transition probability matrix
    P = np.zeros((N_S, N_A, N_S))
    
    # iterate through actions
    
    for u in range(N_A):
        # iterate through states
        for i in range(N_S):
            # converting states to position (x1, x2)
            X = [int((i-i%10)/10), int(i%10)]
            # print("pos:", pos, 'i:', i)
            
            # calculate new pos with action u
            P[i, u, i] = 0.1 # stay
            
            X_prime = X + A[u] # new position with action u
            state = X_prime[0]*10 + X_prime[1] # new state with aciton u 
            
            # check barrier
            if gmap[X[0], X[1]] == 0:
                P[i, u, :] = 0
                P[i, u, i] = 1
            # check for goal state
            elif r_m[X[0], X[1]] == 10:
                P[i, u, :] = 0
                P[i, u, i] = 1
            # check if new pos inside the map
            elif (X_prime[0] >= 0 & X_prime[0] <= 9) and (X_prime[1] >= 0 & X_prime[1] <= 9) and state <= 99:
                # j = state # desired new state
                P[i, u, state] = 0.7 # correct direction
                
                # going the wrong directions
                # c1 = 0
                # c2 = 0
                if u < 2:
                    X_prime = X + A[2]
                    state = X_prime[0]*10 + X_prime[1]
                    if (X_prime[0] >= 0 & X_prime[0] <= 9 & X[0] < 9) and (X_prime[1] >= 0 & X_prime[1] <= 9) and state <= 99:
                        P[i, u, state] = 0.1
                        # c1 += 1
                        # print(X, A[u], X_prime)   
                        
                    X_prime = X + A[3]
                    state = X_prime[0]*10 + X_prime[1]
                    if (X_prime[0] >= 0 & X_prime[0] <= 9 & X[0] > 0) and (X_prime[1] >= 0 & X_prime[1] <= 9) and state <= 99:
                        
                        P[i, u, state] = 0.1
                        # c2 += 1
                        # print(X, A[u], X_prime)
                    
                    # # edge cases
                    # if c1 == 0 or c2 == 0:
                    #     P[i, u , i] = 0.2
                    
                    # c1 = 0
                    # c2 = 0
                    
                if u > 1:
                    X_prime = X + A[0]
                    state = X_prime[0]*10 + X_prime[1]
                    if (X_prime[0] >= 0 & X_prime[0] <= 9) and (X_prime[1] >= 0 & X_prime[1] <= 9 & X[1]<9) and state <= 99:
                        # print(X, A[u], X_prime)
                        P[i, u, state] = 0.1
                        
                        
                    X_prime = X + A[1]
                    state = X_prime[0]*10 + X_prime[1]
                    if (X_prime[0] >= 0 & X_prime[0] <= 9) and (X_prime[1] >= 0 & X_prime[1] <= 9 & X[1]>0) and state <= 99:
                        P[i, u, state] = 0.1
                        # print(X, A[u], X_prime)
                        
                    # edge cases
                    # if c1 == 0 or c2 == 0:
                    #     P[i, u , i] = 0.2
                    
                    # c1 = 0
                    # c2 = 0
            
                
    return P    

# initialize policy and value
pi = [2 for s in range(N_S)] # going east first
V = np.zeros(N_S)

# Initialize rewardr
# reward map
r_m = -np.ones((10,10))
for i in range(10):
    for j in range(10):
        if i == 0 or j ==0:
            r_m[i, j] = -10
        if i == 9 or j == 9:
            r_m[i, j] = -10
r_m[4, 4:8] = -10
r_m[5, 7] = -10
r_m[7,4:6] = -10
r_m[3:7, 2] = -10
r_m[8, 1] = 10

# initialize transition matrix
P = transition_matrix(S, A, gmap, r_m)

# Initialize R
R = np.zeros((N_S, N_S))

for s in range(N_S):
    for s_new in range(N_S):
        
        # converting states to position (x1, x2)
        X = [int((s_new-s_new%10)/10), int(s_new%10)]
        # print(X)
        # print("pos:", pos, 'i:', i)
        R[s, s_new] = r_m[X[0], X[1]]
        # print(R[s, s_new])
        

# discount factor
gamma = 0.9

# print("Initial Policy", pi)
# print V
# print P
# print R

def plot_gmap(gmap):
    plt.figure()
    im = plt.imshow(gmap.T, origin='lower')
    
    ax = plt.gca();
    plt.title('Grid Map')
    
    # # Major ticks
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_yticks(np.arange(0, 10, 1))
    
    # # Labels for major ticks
    ax.set_xticklabels(np.arange(0, 10, 1))
    ax.set_yticklabels(np.arange(0, 10, 1))
    
    # # Minor ticks
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

############################
is_value_changed = True
iterations = 0
# while is_value_changed:
for i in range(7):
    # is_value_changed = False
    # optimal_policy_found = True
    iterations += 1
    # V[36] = sum([R[36, s1] + (P[36, pi[36], s1] * gamma * V[s1]) for s1 in range(N_S)])
    V[36] = sum([P[36,pi[36],s1] * (R[36, s1] + gamma*V[s1]) for s1 in range(N_S)])
    # print(V[36])
    # # value iteration
    for s in range(N_S):
    
    #     # V[s] = sum([P[s,pi[s],s1] * (R[s, s1] + gamma*V[s1]) for s1 in range(N_S)])
        V[s] = sum([P[s,pi[s],s1] * (R[s, s1] + gamma*V[s1]) for s1 in range(N_S)])
        # V[s] = sum([R[s, s1] + (P[s, pi[s], s1] * gamma * V[s1]) for s1 in range(N_S)])
        # print("state", s,"V[s]:", V[s])
    
    # for s in S:

    #         # Compute state value
    #         val = R[s]  # Get direct reward
    #         for s_next in S:
    #             val += P[s, pi[s], s_next] * (gamma * V[s_next])  # Add discounted downstream values

    #         # Update maximum difference
    #         # max_diff = max(max_diff, abs(val - V[s]))

    #         V[s] = val 
    
    
    
    # policy iteration
    for s in range(N_S):
        qf = V[s]
        # print("State", s, "q_best", q_best)
        for a in range(N_A):
            q = sum([P[s, a, s1] * (R[s, s1] + gamma * V[s1]) for s1 in range(N_S)])
            # q = sum([R[s, s1] + (P[s, a, s1] * gamma * V[s1]) for s1 in range(N_S)])
            # print('q_sa', q_sa, 'q_best', q_best)
            if q > qf:
                # print("State", s, ": q", q, "qf", qf)
                pi[s] = a
                qf = q
                # optimal_policy_found = False
                # is_value_changed = True
                
    
    # for s in S:

    #    val_max = V[s]
    #    for a in A:
    #        val = R[s]  # Get direct reward
    #        for s_next in S:
    #            val += P[s, a, s_next] * (gamma * V[s_next])  # Add discounted downstream values

    #        # Update policy if (i) action improves value and (ii) action different from current policy
    #        if val > val_max and pi[s] != a:
    #            pi[s] = a
    #            val_max = val
               # optimal_policy_found = False
    
############################################

    # convert pi from (100, 1) to (10, 10)           
    policy = np.zeros((10,10))
    for i in range(N_S):
        x1 = int((i-i%10)/10)
        x2 = int(i%10)
        policy[x1, x2] = pi[i]
    
    # convert cost from (100, 1) to (10, 10)
    cost = np.zeros((10,10))
    for i in range(N_S):
        x1 = int((i-i%10)/10)
        x2 = int(i%10)
        cost[x1, x2] = V[i]
    
    
    # print("Iterations:", iterations)
    # print("Policy now: \n", policy)
    # print("cost: ", cost)
    # if optimal_policy_found:
    #     break

# plotting cost
plt.figure()
im = plt.imshow(cost.T, origin='lower')



ax = plt.gca();
plt.title('Cost')
for (j,i),label in np.ndenumerate(cost.T):
    ax.text(i,j,int(label),ha='center',va='center')



# # Major ticks
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 10, 1))

# # Labels for major ticks
ax.set_xticklabels(np.arange(0, 10, 1))
ax.set_yticklabels(np.arange(0, 10, 1))

# # Minor ticks
ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
ax.set_yticks(np.arange(-.5, 10, 1), minor=True)

# Gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    

for i in range(3, 10):
    for j in range(6, 10):
        dx = i + a[pi[10*i+j]][0]
        dy = j + a[pi[10*i+j]][1]
        plt.arrow(i, j, dx, dy)
# print("Final policy", pi)
# print(V)
# print(policy)
    