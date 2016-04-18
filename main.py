"""
Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

/Author/: "Shishira R Maiya"
 
"""
from numpy import*
from matplotlib import*
from pylab import*
from math import*

""" Extract data & plot """
data = loadtxt('data.txt')
x = data[:,0]
y = data[:,1]

scatter(x,y,marker='x',c='g')
xlabel('Sand Grain size in beaches of Japan')
ylabel('Probability of finding Wolf spider')

m = y.size
it = ones(shape = (m,2))        # Create it of m X 2 with ones
it[:,1]=x                       # In that replace col 2 with x
temp_x = x                      # we need temp_x , with m X 1 values ie without the collumn of ones
x = it                          # x has col 1 of ones & col 2 with values

j =[]

def sigmoid(z):                          # The sigmoid function used
    return 1.0/(1+e**(1.0*-z))

def compute_cost(x,y,theta):             # To calculate the cost function. Here theta changes in every call ,x and y remain same.     
    h = sigmoid(x.dot(theta).flatten())
    for i in h:
        h = log(i)
        p = log(1-i)
    k = -y.dot(h) - ((1-y).dot(p))
    j = -(1.0/m) * k.sum()
    return j 

def grad(x,y,itera,alpha):
    
    theta = zeros(shape=(2,1))          # Here no of features = 1. So, only two parameters, theta0 & theta 1
    
    for i in range(1,itera):
    
        h = sigmoid(x.dot(theta).flatten())   
        k = (h - y)
        
        temp0 = theta[0][0] - (alpha/m) * k.sum() * 1             # here x0 is [111...1], so multipy by 1
        temp1 = theta[1][0] - (alpha/m) * (k.dot(temp_x)).sum()   # temp_x is x with single rows
        
        if i > 1000:                                              # usually min iterations that are taken = 1000  
            if round(theta[0][0],6) == round(temp0,6):            # if they are equal (here,when precision = 6) ,it means derivative = 0
                print "Exit at",i                                 # The loop exits after precision of 7
                break                                               
        theta[0][0] = temp0                                       # Simultaneous update of theta0 & theta1         
        theta[1][0] = temp1                 
        
        j.append(compute_cost(x,y,theta))                         # just to verify if cost function is decreasing in every iteration
        
    return theta,j        
        
t,j = grad(x,y,150000,0.01)            
print " The values of the parameters "
print '\n'.join(map(str,t))                                      # Will print theta, the parameters

""" The predictions """ 
y_pred = sigmoid(t[0][0] + temp_x.dot(t[1][0]))                  # sigmoid(theta0 + theta1 * x) will be the predictions

print " The predicted values are "
for i in range(m):
    print temp_x[i],'=>',round(y_pred[i],4)
 
plot(x,y_pred,'k-')
show() 
 
print " \nThank You for using the model!\n"                                                        


