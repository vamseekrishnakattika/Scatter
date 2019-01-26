import numpy as np
import sys

# this example insists on 4 arguments
if len(sys.argv) != 5 :
  print(sys.argv[0], "takes 4 arguments. Not ", len(sys.argv)-1)
  sys.exit()

dataFile=open(sys.argv[1],"r")
labelsFile=open(sys.argv[2],"r")
vectorsFile=sys.argv[3]
reducedDataFile=sys.argv[4]

data=np.loadtxt(dataFile,delimiter=',')
y=np.loadtxt(labelsFile)

# another method to calculate B = X X' = sum xi xi'
n,m = data.shape

# calculate mu = mean xi
mean = np.mean(data, axis=0)

# calculate B1,m1,mu1  over xi where yi==1

listVal = y.tolist()

j = set(listVal)
j = list(j)
s =[]
mem = []
for l in j:
    s.append(np.zeros((1,m)))
    mem.append(0)

for l in range(len(j)):
    for i, xt in enumerate(data) :
        if y[i] == j[l] :
            s[l] += xt
            mem[l] += 1

mu = []
for l in range(len(j)):
    mu.append(s[l]/mem[l])


Sc = []
for l in j:
    Sc.append(np.zeros((m,m)))

for l in range(len(j)):
    for i, xt in enumerate(data) :
        if y[i] == j[l] :
            ct = xt - mu[l]
            Sc[l] += np.dot(ct.T,ct)

W = np.zeros((m,m))
for l in range(len(j)):
    W += Sc[l]
    print("S"+str(l+1),"=",Sc[l])
print("W = ",W)

xmu = []
for l in range(len(j)):
    xmu.append(mu[l]-mean)

ymu = []
for l in range(len(j)):
    ymu.append(np.dot(xmu[l].T,xmu[l]))

B = np.zeros((m,m))

for l in range(len(j)):
    B += mem[l]*ymu[l]

print("B = ",B)

Winv = np.linalg.inv(W)

C = np.dot(Winv,B)

print("C = ",C)

evals,evecs = np.linalg.eig(C)

idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:,idx]

r = 2
V_r = evecs[:,:r]

reducedData = np.dot(data,V_r)

np.savetxt(vectorsFile, V_r.T, delimiter=',')
np.savetxt(reducedDataFile, reducedData, delimiter=',')

r = 1
V_r = evecs[:,:r]

final1=np.dot(V_r.T,C)
final2=np.dot(final1,V_r)
print("Maxmum value of ratio of between class to within class scatter:",final2)