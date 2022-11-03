#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:06:52 2022

@author: nicholaskwok
"""
############################### Importing Modules ##########################################

import numpy as np 

import matplotlib.pyplot as plt #Importing plotting module to create 3D Surface Plot

from matplotlib import cm #Importing colour map and aesthetic features

from matplotlib.animation import FuncAnimation #Importing animation module for 3D Surface Plot

############################### Introduction ##########################################

''' 
The multiasset Black Scholes Equation is solved numerically by using an ADI (Alternating Direction Implicit) method 
The multiasset Black Scholes Equation models the time progression of the price of an options strategy 
based on two risky assets, in this case, S1 and S2. 
S1 and S2 are two equities (also known as stocks) and have characteristics, such as volatility
These two equities are correlated with each other, denoted by rho. 
Various options strategies can be used and are coded into the initial conditions
S1, S2, (x, y axis) are the independent space variables
t, is the independent time variable
u is the dependent variable (z axis), which represents the price or value of the option strategy. 
Boundary conditions are that the second derivative (uxx, uyy) is = 0. 
'''

############################### Defining Constants ##########################################

''' 
The constants, independent and dependent variables are defined as follows:

S1 and S2 are two indepdendent variables which represents two equities    

Sig1 and Sig 2 represents volatility of S1 and S2 respectively

Nx, Ny define the number of intervals for the "space" domain for S1 and S2 respectively

r represents the risk free interest rate

rho denotes the correlation between equities S1 and S2, ranges from -1 to 1

K1 and K2 represent the strike price of S1 and S2 respectively

T is the maturity time (the length of time that the function runs for)

deltaT is the time step

c is the cash reward for the Cash or Nothing strategy

'''

#Volatilities
sig1 = 0.1
sig2 = 0.3

#Determining number of space intervals
Nx = 25
Ny = 25

#Defining upper and lower bounds of space intervals
#S1 and S2 cannot be negative. 
#Maximum upperbound of 200 because strike prices K1 and K2 are ~ 100-150
Lb_S1 = 1 
Lb_S2 = 1

Ub_S1 = 200
Ub_S2 = 200

r = 0.05 #Risk free interest rate 

deltaT = 0.005 #Time step 

rho = 1 #Correlation number

#Strike Prices

K1 = 100
K2 = 140

c = 10 #Cash reward for Cash or Nothing Option 

T = 0.5 #Maturity time

############################### Creating domains ##########################################

#Discretizing S1 and S2 domain (x, y respectively)
#Minimum of 1 because S1 and S2 cannot be negative
x = np.linspace(Lb_S1, Ub_S1, Nx)
y = np.linspace(Lb_S2, Ub_S2, Ny)

############################### Helper Functions ##########################################

''' 
Helper functions are used to make script more readable

5 helper functions are created:
        
    update_plot: helps create the frames for the animation
        
        Takes in three arguments: frame_number, zarray, plot
        Annimations consist of frames, and this function updates the plot based on 
        which frame is desired and hence is related to which z-array is used 
        Z-array corresponds to the specific time interval
        
    alpha_coeff: calculates alpha for tridiagonal matrix
    
        Takes in four arguments: i, sig, S, h
        i represents the ith iteration so that alpha is calculated for the right point
        sig represents either sig1 or sig2, depending on S1 and S2
        S represents either S1 or S2, depending which is being evaluated
        h is the space step size
        
    beta_coeff: calculates beta for tridiagonal matrix
    
        Same arguments as alpha_coeff
        
    gamma_coeff: calculates gamma for tridiagonal matrix
    
        Same arguments as alpha_coeff
        
    TDMA: solves tridiagonal matrix in a memory efficient manner (Thomas Algorithm)
        
        Takes in four arguments: ld, md, td, arr
        ld refers to lower diagonal of the tridiagonal matrix
        md refers to middle diagonal of the tridiagonal matrix
        td refers to top diagonal of the tridiagonal matrix
        arr is the array which it solves it with
'''

def update_plot(frame_number, zarray, plot):
    plot[0].remove() #Removes previous plot for animation
    #Plots new surface for a new "frame"
    #Linewidth, rstride, cstride, cmap are aesthetic features
    plot[0] = ax.plot_surface(X, Y, A[frame_number,:,:],linewidth=0,rstride=1, cstride=1,cmap="coolwarm")
    
    return plot

def alpha_coeff(i, sig, S, h): #General calculation for alpha, calculates for both S1 and S2
    return -1*(sig*S[i])**2/(4*h**2) 

def beta_coeff(i, sig, S, h): #General calculation for beta, calculates for both S1 and S2
    return 1/deltaT + (sig*S[i])**2/(2*h**2) + (r*S[i]/(2*h))+r/2

def gamma_coeff(i, sig, S, h): #General calculation for gamma, calculates for both S1 and S2
    return -1*(sig*S[i])**2/(4*h**2) - r*S[i]/(2*h)

def TDMA(ld, md, td, arr): #Arguments are the diagonals of the matrix
    
    n = len(md) #Length of the main diagonal
    q = 0.0
    sol = np.zeros(n, dtype = float) #Initialise solution array
    
    #Copy array in order to avoid changing the original tridiagonal matrix
    
    lower = np.copy(ld)
    middle = np.copy(md)
    top = np.copy(td)
    array = np.copy(arr)
    
    #Elimination step (removes sub-diagonal elements, alpha)

    for k in range(1, n):
        
        q = lower[k]/middle[k-1]
        
        middle[k] = middle[k] - top[k-1]*q
        
        array[k] = array[k] - array[k-1]*q
    
    
    q += array[n-1]/middle[n-1]
    
    sol[n-1] = q
    
    #Back Substitution and solves
    
    for k in range(n-2, -1, -1):
        
        q = (array[k] - top[k]*q)/middle[k]
        
        sol[k] = q 
    
    return sol #Outputs the solution to the linear system of equations

############################### Main Solver ##########################################

''' 
Main Solver consists of two parts

1) Setting and creating initial conditions 

2) Generating tridiagonal matrix
    
3) Executing ADI method

Note that tridiagonal matrix does not vary with time. 
    
ADI method contains two processes:
    
    1) half time step is taken implicitly in x and explicitly in y
        
    2) second half time step is taken implicity in y and explicitly in x 

'''


def BlackScholesADI(sig1, sig2, S1, S2, Nx, Ny, r, deltaT, rho, K1, K2, T, c):
    
    '''
    Black Scholes ADI takes in constants defined above
    Defining basic variables h (space step) and t_int (time step)
    
    3D Master Matrix U houses all solutions for every time step
    
    '''
    
    h = S1[1] - S1[0] #Calculating space step
    
    t_int = int(T/deltaT) #Converts into integer to avoid looping issues
    
    #Defining empty 3D matrix which contains solutions for each time interval (master matrix)
    
    U = np.zeros((t_int + 1, Nx, Ny)) #U[0,:,:] represents the solution for time interval 0
    
    
    ################################ STEP 1 ###################################
    
    '''
    Initialising temporary matrix, u and populating it with initial conditions
    Initial conditions depend on option strategy
    Options strategies may consist of a combination of shorting or longing
    call or put options
    
    Three options strategies are precoded here as intial conditions. 
    
    Simplest option strategy is "Cash Or Nothing":
        - If Stock 1 and Stock 2 surpass their strike price, there is a fixed cash payoff
        - Else, cash payoff = 0 
        - Bullish strategy
        
    Another option strategy is the "Two Asset Call Option":
        - Describes a call option for two assets
        - If Stock 1 or Stock 2 surpass their strike price, the payoff
        is the maximum price of either stock at maturity minus the strike price
        - Bullish strategy
    
    "Butterfly Spread Option" for two assets consists of:
        - 2 long call options, stike price K1 and K2
        - 2 short call options, strike price 0.5*(K1 + K2)
        - Market neutral hedging strategy
    '''
    
    u = np.ndarray((Ny,Nx)) #2D temporary matrix which contains the solution for a particular time step
    
    #Iterate through the matrix and assign values based on options strategy
    for i in range(Ny): 
        
        for j in range(Nx):
            
            ######## Represents Call Option ##########
            
            # u[i,j] = max(x[j] - K1, y[i] - K2, 0)
           
            ######## Represents Butterfly Spread ##########
           
            s = max(S1[j], S2[i])
            
            K = 0.5*(K1+K2)
            
            u[i,j] = max(s-K1, 0) -2*max(s-K, 0) + max(s-K2, 0)
            
            ######## Represents Cash or Nothing ##########
            
            # if x[j]>K1 and y[i]>K2:
                
            #     u[i,j] = c
            
            # else:
                
            #     u[i,j] = 0 
    
    U[0,:,:] = u #Append intitial condition into master matrix 
    
    ################################ STEP 2 ###################################
    
    '''
    Generating two tridiagonal matrices. 
    One tridiagonal matrix to solve S1 (x domain) implicitly
    Another tridiagonal matrix to solve S2 (y domain) implicitly
    Consider lower diagonal, middle diagonal, top diagonal seperately in order to utilise TDMA
    '''
    
    ########## Generating S1 (x domain) Tridiagonal Matrix ##########
    
    '''
    Following formula shown in report, resembling the pattern for the matrix
    '''
    
    #Initialise empty middle diagonal matrix for x domain (S1) 
    mdx = np.zeros(Nx, dtype = float)
    ldx = np.zeros(Nx, dtype = float)
    tdx = np.zeros(Nx, dtype = float)
    
    #Apply end and beginning condiitons if necessary, according to formula
    
    mdx[0] = 2*alpha_coeff(0, sig1, S1, h) + beta_coeff(0, sig1, S1, h)
    mdx[-1] = beta_coeff(-1, sig1, S1, h) + 2*gamma_coeff(-1, sig1, S1, h)
    
    ldx[-1] = alpha_coeff(-1, sig1, S1, h) - gamma_coeff(-1, sig1, S1, h)
    
    tdx[0] = gamma_coeff(0, sig1, S1, h) - alpha_coeff(0, sig1, S1, h)
    
    #Populating arrays to fill in middle values
    
    for j in range(1, Ny-1):
        
        for i in range(1, Nx-1):
            
            ldx[i] = alpha_coeff(i, sig1, S1, h)
            mdx[i] = beta_coeff(i, sig1, S1, h)
            tdx[i] = gamma_coeff(i, sig1, S1, h)
    
    ########## Generating S2 (y domain) Tridiagonal Matrix ##########

    '''
    Note that S2 Tridiagonal Matrix is slightly different to S1 Tridiagonal Matrix
    Due to different Sig values
    '''
    
    #Initialise empty middle diagonal matrix for y domain (S2) 

    mdy = np.zeros(Ny, dtype = float)
    ldy = np.zeros(Ny, dtype = float)
    tdy = np.zeros(Ny, dtype = float)
    
    #Apply end and beginning condiitons if necessary, according to formula
    
    mdy[0] = 2*alpha_coeff(0, sig2, S2, h) + beta_coeff(0, sig2, S2, h)
    mdy[-1] = beta_coeff(-1, sig2, S2, h) + 2*gamma_coeff(-1, sig2, S2, h)
    
    ldy[-1] = alpha_coeff(-1, sig2, S2, h) - gamma_coeff(-1, sig2, S2, h)
   
    tdy[0] = gamma_coeff(0, sig2, S2, h) - alpha_coeff(0, sig2, S2, h)
    
    #Populating arrays to fill in middle values
   
    for i in range(1, Nx-1):
        
        for j in range(1, Ny-1):
            
            ldy[j] = alpha_coeff(j, sig2, S2, h)
            mdy[j] = beta_coeff(j, sig2, S2, h)
            tdy[j] = gamma_coeff(j, sig2, S2, h) 
    
    ################################ STEP 3 ###################################
    
    '''
    ADI Method is executed every time step
    Executing the ADI method requires two processes
    '''

    for t in range(1, t_int + 1): #Time iteration
        
        u = U[t-1] #Updates temporary matrix u with the solution from previous time interval

        ########## Process 1 ##########
        #Solving implictly for x domain (S1) but explicitly in y domain (S2) 
        
        #Initialise a new matrix, sol_u to contain solutions for this particular time step
        
        sol_u = np.zeros((Ny, Nx)) 
        
        for j in range (Ny):
            
            fij = [] #Creating fij array to implement ADI method for n-halfth interval
            
            for i in range(Nx):
                    
                #Populating array fij, derived from finite difference methods, which depends on each time interval
                #Implement boundary conditions to account for edge and corner cases around boundaries
                
                if i==0 and j==0:
                    
                    fij+=[u[i,j]/deltaT + 1/4*(sig2*S2[j])**2*((u[i,j+1]) - 2*u[i,j] + u[i,j-1])/h**2
+1/2*r*S2[j]*(u[i,j+1] - u[i,j])/h + 1/2*rho*sig1*sig2*S1[i]*S2[j]*(u[i+1,j+1] 
 + (4*u[i,j] - 2*u[i+1,j] - 2*u[i,j+1] + u[i+1,j+1]) - (2*u[i,j+1] - u[i+1,j+1]) - u[i+1,j-1])/(4*h**2)]
                
                elif i== Nx-1 and j== 0:
                    
                    fij+=[u[i,j]/deltaT+1/4*(sig2*S2[j])**2*((u[i,j+1])-2*u[i,j]+(2*u[i,j]-u[i,j+1]))/h**2
            +1/2*r*S2[j]*(u[i,j+1]-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*((2*u[i,j+1]-u[i-1,j+1])+(2*u[i-1,j]-u[i-1,j+1])-u[i-1,j+1]-(4*u[i,j]-6*u[i,j+1]+4*u[i,j+2]-u[i,j+3]))/(4*h**2)]

                elif i==0 and j==Ny-1:
                    
                    fij+=[u[i,j]/deltaT+1/4*(sig2*S2[j])**2*((2*u[i,j]-u[i,j-1])-2*u[i,j]+u[i,j-1])/h**2
            +1/2*r*S2[j]*((2*u[i,j]-u[i,j-1])-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*((2*u[i+1,j]-u[i+1,j-1])+(2*u[i,j-1]-u[i+1,j-1])-(4*u[i,j]-6*u[i+1,j]+4*u[i+2,j]-u[i+3,j])-u[i+1,j-1])/(4*h**2)]
                
                elif i==Nx-1 and j==Ny-1:
                    
                    fij+=[u[i,j]/deltaT+1/4*(sig2*S2[j])**2*((2*u[i,j]-u[i,j-1])-2*u[i,j]+u[i,j-1])/h**2
            +1/2*r*S2[j]*(2*u[i,j]-u[i,j-1])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*((4*u[i,j]-2*u[i,j-1]-2*u[i-1,j]+u[i-1,j-1])+u[i-1,j-1]-(2*u[i-1,j]-u[i-1,j-1])-(2*u[i,j-1]- u[i-1,j-1]))/(4*h**2)]
                                                                           
                elif i==0:
                    
                    fij+=[u[i,j]/deltaT+1/4*(sig2*S2[j])**2*((u[i,j+1])-2*u[i,j]+u[i,j-1])/h**2
            +1/2*r*S2[j]*(u[i,j+1]-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*(u[i+1,j+1]+(2*u[i,j-1]-u[i+1,j-1])-(2*u[i,j+1]-u[i+1,j+1])-u[i+1,j-1])/(4*h**2)]
                    
                elif i==Nx-1:
                    
                    fij+=[u[i,j]/deltaT+1/4*(sig2*S2[j])**2*((u[i,j+1])-2*u[i,j]+u[i,j-1])/h**2
            +1/2*r*S2[j]*(u[i,j+1]-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*((2*u[i,j+1]-u[i-1,j+1])+u[i-1,j-1]-u[i-1,j+1]-(2*u[i,j-1]-u[i-1,j-1]))/(4*h**2)]
                    
                
                elif j==0:
                    
                    fij+=[u[i,j]/deltaT+1/4*(sig2*S2[j])**2*((u[i,j+1])-2*u[i,j]+u[i,j-1])/h**2
            +1/2*r*S2[j]*(u[i,j+1]-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*(u[i+1,j+1]+(2*u[i-1,j]-u[i-1,j+1])-u[i-1,j+1]-(2*u[i+1,j]-u[i+1,j+1]))/(4*h**2)]
                    
                    
                elif j==Ny-1:
                    
                    fij+=[u[i,j]/deltaT+1/4*(sig2*S2[j])**2*((2*u[i,j]-u[i,j-1])-2*u[i,j]+u[i,j-1])/h**2
            +1/2*r*S2[j]*((2*u[i,j]-u[i,j-1])-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*((2*u[i+1,j]-u[i+1,j-1])+u[i-1,j-1]-(2*u[i-1,j]-u[i-1,j-1])-u[i+1,j-1])/(4*h**2)]
                  
                else:
                    
                    fij+=[u[i,j]/deltaT+1/4*(sig2*S2[j])**2*((u[i,j+1])-2*u[i,j]+u[i,j-1])/h**2 + 
                          1/2*r*S2[j]*(u[i,j+1]-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*(u[i+1,j+1]+u[i-1,j-1]-u[i-1,j+1]-u[i+1,j-1])/(4*h**2)]
            
            #Solve using linear system of equations using TDMA to find solutions for S1 (space x) domain
        
            sol_u[j, :] = TDMA(ldx, mdx, tdx, fij)

        
        #Update temporary matrix u with solutions for n-halfth time step
        
        u = sol_u
        
        ########## Process 2 ##########
        #Solving implictly for y domain (S2) but explicitly in x domain (S1) 
        
        #Initialise a new matrix, sol_u to contain solutions for this particular time step

        sol_u = np.zeros((Ny, Nx))
        
        for i in range (Nx):
            
            gij=[] #Creating gij array to implement ADI method for next n-halfth time interval
            
            for j in range (Ny):
                    
                #Populating array gij, similar to fij but with slightly different formula 
                #Implement boundary conditions  to account for edge and corner cases around boundaries
                
                if i==0 and j==0:
                    
                    gij+=[u[i,j]/deltaT+1/4*(sig1*S1[i])**2*((u[i+1,j])-2*u[i,j]+(2*u[i,j]-u[i+1,j]))/h**2
                          +1/2*r*S1[i]*(u[i+1,j]-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*(u[i+1,j+1]+(4*u[i,j]-2*u[i+1,j]-2*u[i,j+1]+u[i+1,j+1])-(2*u[i,j+1]-u[i+1,j+1])-u[i+1,j-1])/(4*h**2)]
                
                elif i== Nx-1 and j== 0:
                    
                    gij+=[u[i,j]/deltaT+1/4*(sig1*S1[i])**2*((2*u[i,j]-u[i-1,j])-2*u[i,j]+u[i-1,j])/h**2
                          +1/2*r*S1[i]*((2*u[i,j]-u[i-1,j])-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*((2*u[i,j+1]-u[i-1,j+1])+(2*u[i-1,j]-u[i-1,j+1])-u[i-1,j+1]-(4*u[i,j]-6*u[i,j+1]+4*u[i,j+2]-u[i,j+3]))/(4*h**2)]
                                                                         
                elif i==0 and j==Ny-1:
                    
                    gij+=[u[i,j]/deltaT+1/4*(sig1*S1[i])**2*((u[i+1,j])-2*u[i,j]+(2*u[i,j]-u[i+1,j]))/h**2
                          +1/2*r*S1[i]*(u[i+1,j]-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*((2*u[i+1,j]-u[i+1,j-1])+(2*u[i,j-1]-u[i+1,j-1])-(4*u[i,j]-6*u[i+1,j]+4*u[i+2,j]-u[i+3,j])-u[i+1,j-1])/(4*h**2)]
            
                elif i==Nx-1 and j==Ny-1:
                    
                   gij+=[u[i,j]/deltaT+1/4*(sig1*S1[i])**2*((2*u[i,j]-u[i-1,j])-2*u[i,j]+u[i-1,j])/h**2
                         +1/2*r*S1[i]*((2*u[i,j]-u[i-1,j])-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*((4*u[i,j]-2*u[i,j-1]-2*u[i-1,j]+u[i-1,j-1])+u[i-1,j-1]-(2*u[i-1,j]-u[i-1,j-1])-(2*u[i,j-1]- u[i-1,j-1]))/(4*h**2)]
                    
                elif i==0:
                    
                    gij+=[u[i,j]/deltaT+1/4*(sig1*S1[i])**2*((u[i+1,j])-2*u[i,j]+(2*u[i,j]-u[i+1,j]))/h**2
                          +1/2*r*S1[i]*(u[i+1,j]-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*(u[i+1,j+1]+(2*u[i,j-1]-u[i+1,j-1])-(2*u[i,j+1]-u[i+1,j+1])-u[i+1,j-1])/(4*h**2)]
                    
                elif i==Nx-1:
                    
                    gij+=[u[i,j]/deltaT+1/4*(sig1*S1[i])**2*((2*u[i,j]-u[i-1,j])-2*u[i,j]+u[i-1,j])/h**2
                          +1/2*r*S1[i]*((2*u[i,j]-u[i-1,j])-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*((2*u[i,j+1]-u[i-1,j+1])+u[i-1,j-1]-u[i-1,j+1]-(2*u[i,j-1]-u[i-1,j-1]))/(4*h**2)]
                    
                
                elif j==0:
                    
                    gij+=[u[i,j]/deltaT+1/4*(sig1*S1[i])**2*((u[i+1,j])-2*u[i,j]+u[i-1,j])/h**2
                          +1/2*r*S1[i]*(u[i+1,j]-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*(u[i+1,j+1]+(2*u[i-1,j]-u[i-1,j+1])-u[i-1,j+1]-(2*u[i+1,j]-u[i+1,j+1]))/(4*h**2)]
            
                    
                elif j==Ny-1:
                    
                   gij+=[u[i,j]/deltaT+1/4*(sig1*S1[i])**2*((u[i+1,j])-2*u[i,j]+u[i-1,j])/h**2
                         +1/2*r*S1[i]*(u[i+1,j]-u[i,j])/h+1/2*rho*sig1*sig2*S1[i]*S2[j]*((2*u[i+1,j]-u[i+1,j-1])+u[i-1,j-1]-(2*u[i-1,j]-u[i-1,j-1])-u[i+1,j-1])/(4*h**2)]
           
                  
                else:
                    
                    gij+=[u[i,j]/deltaT+1/4*(sig1*S1[i])**2*((u[i+1,j])-2*u[i,j]+u[i-1,j])/h**2+1/2*r*S1[i]*(u[i+1,j]-u[i,j])/h
                    +1/2*rho*sig1*sig2*S1[i]*S2[j]*(u[i+1,j+1]+u[i-1,j-1]-u[i-1,j+1]-u[i+1,j-1])/(4*h**2)]
            
            
            #Solve using linear system of equations using TDMA to find solutions for S2 (space y) domain
            
            sol_u[:, i] += TDMA(ldy, mdy, tdy, gij)
             
        #Update temporary matrix u with solutions for full time step and append into master matrix U
            
        U[t]=sol_u #Keep on looping through each time interval to populate master matrix U
        
    return U

########################### Plotting and Calling Function #######################################

#Calling function with desired constants
A = BlackScholesADI(sig1, sig2, x, y, Nx, Ny, r, deltaT, rho, K1, K2, T, c) 

X,Y =np.meshgrid(x,y) #Creating grid of nodes for plotting

iterations = int(T/deltaT)+1 #Calculating number of iterations for animation

# Plotting figures using Matplotlib library
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.invert_yaxis()
ax.set_zlim(0,20)

ax.set_xlabel('Stock price 1')

ax.set_ylabel('Stock price 2')

ax.set_zlabel('Option price')

plot = [ax.plot_surface(X, Y, A[0,:,:],linewidth=0,rstride=1, cstride=1)]

ani = FuncAnimation(fig, update_plot, iterations, fargs=(A, plot), interval= 10)

plt.show()

# Saving animation

ani.save('BlackScholes2Asset.gif', fps = 60, dpi = 75)

print("Check folder for animation gif!")
print("Or to view the animation, if on Spyder, change preferences:")
print("Under IPython Console and Graphics, Set Graphics Backend to Automatic")