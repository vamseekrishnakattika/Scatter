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
mu = np.mean(data, axis=0)

s1 = np.zeros((1,m))
m1 = 0

for i, xt in enumerate(data) :
    if y[i] == 1 :
        s1 += xt
        m1 += 1
        
mu1 = s1/m1

s2 = np.zeros((1,m))
m2 = 0

for i, xt in enumerate(data) :
    if y[i] == 2 :
        s2 += xt
        m2 += 1
        
mu2 = s2/m2

s3 = np.zeros((1,m))
m3 = 0

for i, xt in enumerate(data) :
    if y[i] == 3 :
        s3 += xt
        m3 += 1
        
mu3 = s3/m3

# Calculate C1, the covariance matrix over xi where yi = 1
S1 = np.zeros((m,m))
for i, xt in enumerate(data) :
    if y[i] == 1 :
        ct = xt - mu1
        S1 += np.dot(ct.T,ct)

# Calculate C2, the covariance matrix over xi where yi = 2
S2 = np.zeros((m,m))
for i, xt in enumerate(data) :
    if y[i] == 2 :
        ct = xt - mu2
        S2 += np.dot(ct.T,ct)

# Calculate C3, the covariance matrix over xi where yi = 3
S3 = np.zeros((m,m))
for i, xt in enumerate(data) :
    if y[i] == 3 :
        ct = xt - mu3
        S3 += np.dot(ct.T,ct)

W=S1+S2+S3

xmu1=mu1-mu
xmu2=mu2-mu
xmu3=mu3-mu

ymu1=np.dot(xmu1.T,xmu1)
ymu2=np.dot(xmu2.T,xmu2)
ymu3=np.dot(xmu3.T,xmu3)

B=m1*ymu1+m2*ymu2+m3*ymu3

Winv = np.linalg.inv(W)

C = np.dot(Winv,B)

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