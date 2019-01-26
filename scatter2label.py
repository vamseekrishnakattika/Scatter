import numpy as np
import sys

dataFile=open(sys.argv[1],"r")
labelsFile=open(sys.argv[2],"r")
vectorsFile=sys.argv[3]
reducedDataFile=sys.argv[4]

data=np.loadtxt(dataFile,delimiter=',')
y=np.loadtxt(labelsFile)

n,m = data.shape

mean = np.mean(data, axis=0)

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
evals,evecs = np.linalg.eig(B)
idx = np.argsort(evals)[::-1]

# sort in reverse order
evals = evals[idx]
evecs = evecs[:,idx]

r = 2
V_r = evecs[:,:r]

reducedData = np.dot(data,V_r)

np.savetxt(vectorsFile, V_r.T, delimiter=',')
np.savetxt(reducedDataFile, reducedData, delimiter=',')

r = 1
V_r = evecs[:,:r]

final1=np.dot(V_r.T,B)
final2=np.dot(final1,V_r)
print("Maximum value for between class scatter:",final2)