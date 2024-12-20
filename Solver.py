from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def NACA_4digit(m:float, p:float, x:np.ndarray[float]) -> np.ndarray[float]:
    def naca_4digit(m:float, p:float, x:float) -> float:
        if p == 0:
            return 0
        if 0 <= x <= p:
            return m/p**2 * (2*p*x - x**2)
        else:
            return m/(1 - p)**2 * (1 - 2*p + 2*p*x - x**2)
    return  np.array([naca_4digit(m, p, i) for i in x])

def NACA_4digit_derivative(m:float, p:float, x:np.ndarray[float]) -> np.ndarray[float]:
    def naca_4digit_derivative(m:float, p:float, x:float) -> float:
        if 0 <= x <= p:
            return m/p**2 * (2*p - 2*x)
        else:
            return m/(1 - p)**2 * (2*p - 2*x)
    return  np.array([naca_4digit_derivative(m, p, i) for i in x])

def symetric_chamberline(x:np.ndarray[float]) -> np.ndarray[float]:
    return np.zeros(len(x))

def symetric_chamberline_derivative(x:np.ndarray[float]) -> np.ndarray[float]:
    return np.zeros(len(x))


def create_panels (f: Callable[[float], float], N: int, linear_spacing: bool = True, df: Callable[[float], float]|None = None) -> None:
    """
    Creates nodes, Discrete vortex locations and collonation/control points
    input f = function of camber line
    input N = number of panels
    linear_spacing = True if panels are equally spaced false not implemented
    """


    Xn = np.linspace(0,1, N+1)
    if not linear_spacing: # cosine spacing
        Xn = (1 - np.cos(Xn*np.pi)) / 2
    Xv = Xn[:N] + (Xn[1:N+1] - Xn[:N]) / 4
    Xc = Xn[:N] + (Xn[1:N+1] - Xn[:N]) * 3 / 4
    Yn = f(Xn)
    Yv = f(Xv)
    Yc = f(Xc)

    if df:
        # normal vector of chamberline
        dx = np.ones(N)
        dy = df(Xc)
        v_normal = np.column_stack((-dy, dx))

    else:
        # normal vector of panel
        dx = Xn[1:] - Xn[:-1]
        dy = Yn[1:] - Yn[:-1]
        v_normal = np.column_stack((-dy, dx))

    return Xn, Yn, Xv, Yv, Xc, Yc, v_normal

# VOR2D(XC(I),ZC(I),X(J),Z(J),1.0,U,W)

def unit_influence(x:float, y:float, xj:float, yj:float) -> np.ndarray[float, float]:
    """
    Computes the influence of a unit vortex placed at x,y at point P = (xj,yj)
    returns v = (u, w)
    """
    Dx = xj - x
    Dy = yj - y
    r2 = Dx * Dx + Dy * Dy
    C = 0.5 / np.pi / r2

    v = np.dot(np.array([[0.0, 1.],
                  [-1., 0.]]), np.array([Dx, Dy])) * C

    return v

def assemble_A(Xv, Yv, Xc, Yc, v_normal):
    """Assembles the influence matrix"""
    N = Xc.shape[0]
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            A[i, j] = np.dot(unit_influence(Xv[j], Yv[j], Xc[i], Yc[i]), v_normal[j, :])
    return A


def assemble_b(alpha, v_normal):
    """
    Assembles the b vector
    Contains the  negative of the freestream contribution to the normal velocity at eachpanel

    alpha = angle of attack in radians
    v_normal = normal vector of each collonation point
    """
    U = np.cos(alpha)
    W = np.sin(alpha)
    V = np.array([[U],
                  [W]])
    RHS = np.dot(v_normal, -V)
    return RHS

def dCp_Cl(Gamma, Xv, Yv, dc, alpha):
    """Computing the pressure distribution and the lift coefficient
    RHS uses unit freestream velocity and unit chord simplifying Kutta-Joukowski theory
    To 2 times the sum of vortex strength"""
    dCp = 2*Gamma.flatten()/dc
    Cl = 2 * np.sum(Gamma)
    dcn = 2*Gamma * np.cos(alpha)
    dct = 2*Gamma * np.sin(alpha)
    dcm = - dcn.flatten() * (Xv - 1/4) + dct.flatten() * Yv  # quarter chord
    CM = np.sum(dcm)
    # print(f'alpha = {alpha*180/np.pi}, cm = {CM}')
    # plt.plot(Xv, dcm, color = 'red')
    # plt.show()

    return dCp, Cl, CM

def analytical(alpha, N):
    """Analytical solution of the thin flat airfoil"""
    x = np.linspace(0.01, 1, N)
    CL = 2 *np.pi * np.sin(alpha)
    Cm = -np.pi/2 * np.sin(alpha) - CL/4
    dCp = 4.0 * np.sqrt((1 - x) / x ) * alpha
    return CL, Cm, dCp, x

def read_cp_data(file_path: str):
    """
    read xfoil cp data from file
    """
    data  = np.fromfile(file_path, sep='\t')
    x     = data[0::2]
    cp    = data[1::2]
    return x, cp

def Dcp(x, cp):
    indexeadingedge = 0
    x0 = 1.1
    for i,xi in enumerate(x):
        if xi > x0:
            break
        x0 = xi
    cpupper = cp[0:i+1]
    cplower  = cp[i:]
    cplower =  cplower[::-1]
    xupper = x[0:i+1]
    xlower = x[i:]
    # linear interpolation between data points upper surface to match x locations of lower surface
    xlower = xlower[::-1]
    cpupnew = np.zeros(len(xlower))

    # To compute delta cp data locaions need to align thus data locations of cp upper are shifted to the same location as cp lower
    for i, xi in enumerate(xlower):
        x1 = 0
        x2 = 1
        equal = False
        for j, xj in enumerate(xupper):

            if xj==xi:
                cpupnew[i] = cpupper[j]
                equal = True
                break
            elif xj < xi and xj > x1:
                x1 = xj
                indx1 = j
                cp1 = cpupper[j]

            elif xj > xi and xj < x2:
                x2 = xj
                indx2 = j
                cp2 = cpupper[j]
        if not equal:
            # linear interpolation to get cp at new location
            a = (cp1-cp2)/(x1-x2)
            dx = xi - x1
            cpupnew[i] = cp1 + a * dx


    dcp = np.array(cplower) - cpupnew
    return dcp, xlower


def Comparison(x1, x2, cp1, cp2,):
    plt.rc('font', size=15)
    plt.plot(x1, cp1, "-", color = 'blue', label = 'Thin airfoil')
    plt.plot(x2, cp2, "--" , color = 'black', label = 'X-foil invsicid')
    plt.xlabel('x/c')
    plt.ylabel(r'$\Delta Cp$')
    plt.legend()
    plt.gca().invert_yaxis()  # This line inverts the y-axis
    plt.show()

def numerical_routine(airfoil, AOA, N):
    Airfoil = airfoil  # Last 2 digits are insignificant or thin airfoils

    m = float(Airfoil[0]) / 100
    p = float(Airfoil[1]) / 10
    alpha = AOA * np.pi / 180


    #
    f = lambda x: NACA_4digit(m, p, x)
    df = lambda x: NACA_4digit_derivative(m, p, x)

    # create panels
    Xn, Yn, Xv, Yv, Xc, Yc, v_normal = create_panels(f, N, linear_spacing=False, df=df)
    dc = Xn[1:] - Xn[:-1]  # panel lenghts

    # create influence matrix
    A = assemble_A(Xv, Yv, Xc, Yc, v_normal)

    # create b vector (right hand side)
    b = assemble_b(alpha, v_normal)

    # solve linear system
    Gamma = np.linalg.solve(A, b)

    # compute delta pressures and lift coefficient
    dCp, Cl, Cm = dCp_Cl(Gamma, Xv, Yv, dc, alpha)

    return Xv, dCp, Cl, Cm



def main():

    Airfoil = '0000' #  Last 2 digits are insignificant or thin airfoils
    m = int(Airfoil[0]) /100
    p = int(Airfoil[1]) / 10
    alpha = 3 * np.pi / 180
    N = 100

    #
    f = lambda x: NACA_4digit(m, p, x)
    df = lambda x: NACA_4digit_derivative(m, p, x)

    # create panels
    Xn, Yn, Xv, Yv, Xc, Yc, v_normal = create_panels(f, N, linear_spacing=False, df=df)
    dc = Xn[1:] - Xn[:-1] # panel lenghts

    # create influence matrix
    A = assemble_A(Xv, Yv, Xc, Yc, v_normal)

    # create b vector (right hand side)
    b = assemble_b(alpha, v_normal)

    # solve linear system
    Gamma = np.linalg.solve(A, b)

    # compute delta pressures and lift coefficient
    dCp, Cl, Cm = dCp_Cl(Gamma, Xv, Yv, dc, alpha)


    if m == 0:
        #analytical results flat plate
        Cl_an, Cm_an, dCp_an, x_an = analytical(alpha, 100)
        plt.plot(x_an, dCp_an, label='analytical')
        print('lift coefficient analytical:', Cl_an)
        print('Cm analytical:', Cm_an)

    # print results
    print('GAMMA: ', Gamma[-1])
    print('lift coefficient numerical:', Cl)
    print('Cm numerical:', Cm)
    plt.plot(Xv, dCp, label='numerical')
    plt.scatter(Xv, dCp)

    plt.legend()
    plt.show()


def convergence(N, airfoil):
    """
    test convergence in Cl_a for increasing number of panels
    """
    m = int(airfoil[0]) / 100
    p = int(airfoil[1]) / 10
    Cla = []
    Nlist = range(1, N+2, 1)
    for n in Nlist:
        f = lambda x: NACA_4digit(m, p, x)
        df = lambda x: NACA_4digit_derivative(m, p, x)
        Xn, Yn, Xv, Yv, Xc, Yc, v_normal = create_panels(f, n, linear_spacing=False, df=df)
        dc = Xn[1:] - Xn[:-1]  # panel lenghts
        alist = []
        cllist = []
        for alpha in range(1, 10,1):
            alpha = alpha/180 * np.pi # from 0-10 deg AOA converted to radians
            A = assemble_A(Xv, Yv, Xc, Yc, v_normal)
            b = assemble_b(alpha, v_normal)
            Gamma = np.linalg.solve(A, b)
            _, Cl, _ = dCp_Cl(Gamma, Xv, Yv, dc, alpha)
            alist.append(alpha)
            cllist.append(Cl)
        a , b = np.polyfit(alist, cllist, 1)
        Cla.append(a)
        # plt.plot(alist, cllist)


    fig, ax1 = plt.subplots()

    color = 'black'
    ax1.set_xlabel('Number of panels')
    ax1.set_ylabel(r'$Cl_{\alpha}$', color=color)
    ax1.plot(Nlist, Cla, label=r'$Cl_{\alpha}$', color=color)
    ax1.axhline(np.pi * 2, color='black', linestyle='dashed', label=r'$2\pi$')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Error', color=color)
    ax2.plot(Nlist, np.array(Cla) - 2 * np.pi, label='error', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.legend(loc='upper left')
    fig.tight_layout()
    plt.show() # plots edited with codeium

def C_alpha(airfoil, N, AOArange):
    Cl = []
    Cm = []
    for AOA in AOArange:
        _,_, Cli, Cmi =numerical_routine(airfoil, AOA, N)
        Cl.append(Cli)
        Cm.append(Cmi)
    return Cl, Cm


def question2a():
    # load xfoil polars
    naca0408 = pd.read_csv('naca0408polars.txt', skiprows=[x for x in  range(0, 12) if x != 10], sep=r'\s+', header=0) # data from xfoil
    naca1408 = pd.read_csv('naca1408polars.txt', skiprows=[x for x in  range(0, 12) if x != 10], sep=r'\s+', header=0) # data from xfoil
    naca2408 = pd.read_csv('naca2408polars.txt', skiprows=[x for x in  range(0, 12) if x != 10], sep=r'\s+', header=0) # data from xfoil
    naca3408 = pd.read_csv('naca3408polars.txt', skiprows=[x for x in  range(0, 12) if x != 10], sep=r'\s+', header=0) # data from xfoil
    naca4408 = pd.read_csv('naca4408polars.txt', skiprows=[x for x in  range(0, 12) if x != 10], sep=r'\s+', header=0) # data from xfoil
    naca5408 = pd.read_csv('naca5408polars.txt', skiprows=[x for x in  range(0, 12) if x != 10], sep=r'\s+', header=0) # data from xfoil

    # calculate thin airfoil polars
    angles = range(-15, 16, 1)
    N = 100
    cl1 , cm1 = C_alpha([0.5, 4], N, angles) # naca (0.5)408
    cl2, cm2 = C_alpha('1408', N, angles) # naca1108
    cl3, cm3 = C_alpha('2408', N, angles) # naca2408
    cl4, cm4 = C_alpha('3408', N, angles) # naca3408
    cl5, cm5 = C_alpha('4408', N, angles) # naca4408
    cl6, cm6 = C_alpha('5408', N, angles) # naca5408

    #plot xfoil
    plt.rc('font', size=15)
    plt.plot(naca0408['alpha'], naca0408['CL'], label = r'0.5%', color = 'blue')
    plt.plot(naca1408['alpha'], naca1408['CL'], label = r'1%', color = 'black')
    plt.plot(naca2408['alpha'], naca2408['CL'], label = r'2%', color = 'red')
    plt.plot(naca3408['alpha'], naca3408['CL'], label = r'3%', color = 'green')
    plt.plot(naca4408['alpha'], naca4408['CL'], label = r'4%', color = 'purple')
    plt.plot(naca5408['alpha'], naca5408['CL'], label = r'5%', color = 'orange')

    # plot thin airfoil polars
    plt.plot(angles, cl1, color = 'blue', linestyle = '--')
    plt.plot(angles, cl2, color = 'black', linestyle = '--')
    plt.plot(angles, cl3, color = 'red', linestyle = '--')
    plt.plot(angles, cl4, color = 'green', linestyle = '--')
    plt.plot(angles, cl5, color = 'purple', linestyle = '--')
    plt.plot(angles, cl6, color = 'orange', linestyle = '--')

    plt.xlabel(r'$\alpha$ [deg]')
    plt.ylabel(r'$C_L$')
    plt.legend()
    plt.title('Effect of max camber NACAx4(08)')
    plt.show()

    plt.plot(naca0408['alpha'], naca0408['CM'], label = r'0.5%', color = 'blue')
    plt.plot(naca1408['alpha'], naca1408['CM'], label = r'1%', color = 'black')
    plt.plot(naca2408['alpha'], naca2408['CM'], label = r'2%', color = 'red')
    plt.plot(naca3408['alpha'], naca3408['CM'], label = r'3%', color = 'green')
    plt.plot(naca4408['alpha'], naca4408['CM'], label = r'4%', color = 'purple')
    plt.plot(naca5408['alpha'], naca5408['CM'], label = r'5%', color = 'orange')

    plt.plot(angles, cm1, color = 'blue', linestyle = '--')
    plt.plot(angles, cm2, color = 'black', linestyle = '--')
    plt.plot(angles, cm3, color = 'red', linestyle = '--')
    plt.plot(angles, cm4, color = 'green', linestyle = '--')
    plt.plot(angles, cm5, color = 'purple', linestyle = '--')
    plt.plot(angles, cm6, color = 'orange', linestyle = '--')

    # adding axis labels
    plt.xlabel(r'$\alpha$ [deg]')
    plt.ylabel(r'$C_m$')
    plt.legend()
    plt.title('Effect of max camber NACAx4(08)')
    plt.show()


def question2b():
    # load xfoil polars
    naca2208 = pd.read_csv('naca2208polars.txt', skiprows=[x for x in range(0, 12) if x != 10], sep=r'\s+',
                           header=0)  # data from xfoil
    naca2308 = pd.read_csv('naca2308polars.txt', skiprows=[x for x in range(0, 12) if x != 10], sep=r'\s+',
                           header=0)  # data from xfoil
    naca2408 = pd.read_csv('naca2408polars.txt', skiprows=[x for x in range(0, 12) if x != 10], sep=r'\s+',
                           header=0)  # data from xfoil
    naca2508 = pd.read_csv('naca2508polars.txt', skiprows=[x for x in range(0, 12) if x != 10], sep=r'\s+',
                           header=0)  # data from xfoil
    naca2608 = pd.read_csv('naca2608polars.txt', skiprows=[x for x in range(0, 12) if x != 10], sep=r'\s+',
                           header=0)  # data from xfoil


    # calculate thin airfoil polars
    angles = range(-15, 16, 1)
    N = 100
    cl1, cm1 = C_alpha('2208', N, angles)  # naca (0.5)408
    cl2, cm2 = C_alpha('2308', N, angles)  # naca1108
    cl3, cm3 = C_alpha('2408', N, angles)  # naca2408
    cl4, cm4 = C_alpha('2508', N, angles)  # naca3408
    cl5, cm5 = C_alpha('2608', N, angles)  # naca4408
    cl6, cm6 = C_alpha('2908', N, angles)  # naca2908

    # plot xfoil
    plt.rc('font', size=15)
    plt.plot(naca2208['alpha'], naca2208['CL'], label=r'$\frac{c}{x} = 0.2$', color='blue')
    plt.plot(naca2308['alpha'], naca2308['CL'], label=r'$\frac{c}{x} = 0.3$', color='black')
    plt.plot(naca2408['alpha'], naca2408['CL'], label=r'$\frac{c}{x} = 0.4$', color='red')
    plt.plot(naca2508['alpha'], naca2508['CL'], label=r'$\frac{c}{x} = 0.5$', color='green')
    plt.plot(naca2608['alpha'], naca2608['CL'], label=r'$\frac{c}{x} = 0.6$', color='purple')


    # plot thin airfoil polars
    plt.plot(angles, cl1, color='blue', linestyle='--')
    plt.plot(angles, cl2, color='black', linestyle='--')
    plt.plot(angles, cl3, color='red', linestyle='--')
    plt.plot(angles, cl4, color='green', linestyle='--')
    plt.plot(angles, cl5, color='purple', linestyle='--')
    # plt.plot(angles, cl6, color='pink', linestyle='--')

    plt.xlabel(r'$\alpha$ [deg]')
    plt.ylabel(r'$C_L$')
    plt.legend()
    plt.title('Effect of camber position NACA2x(08)')
    plt.show()

    plt.plot(naca2208['alpha'], naca2208['CM'], label=r'$\frac{c}{x} = 0.2$', color='blue')
    plt.plot(naca2308['alpha'], naca2308['CM'], label=r'$\frac{c}{x} = 0.3$', color='black')
    plt.plot(naca2408['alpha'], naca2408['CM'], label=r'$\frac{c}{x} = 0.4$', color='red')
    plt.plot(naca2508['alpha'], naca2508['CM'], label=r'$\frac{c}{x} = 0.5$', color='green')
    plt.plot(naca2608['alpha'], naca2608['CM'], label=r'$\frac{c}{x} = 0.6$', color='purple')


    plt.plot(angles, cm1, color='blue', linestyle='--')
    plt.plot(angles, cm2, color='black', linestyle='--')
    plt.plot(angles, cm3, color='red', linestyle='--')
    plt.plot(angles, cm4, color='green', linestyle='--')
    plt.plot(angles, cm5, color='purple', linestyle='--')

    # adding axis labels
    plt.xlabel(r'$\alpha$ [deg]')
    plt.ylabel(r'$C_m$')
    plt.legend()
    plt.title('Effect of camber position NACA2x(08)')
    plt.show()
    pass







if __name__ == '__main__':
    main()
    convergence(100, '2408')
    x, cp = read_cp_data('cp2408.txt')
    dcp1, x1 = Dcp(x, cp)
    x2, dcp2,_,_ = numerical_routine('2408', 0, 400)
    comparison = Comparison(x2, x1, dcp2, dcp1) # cp plot
    angles = np.linspace(-15, 15, 20)
    Cl, Cm = C_alpha('2408', 100, angles) # cl , cm data for thin airfoil
    df = pd.read_csv('naca2408polars.txt', skiprows=[x for x in  range(0, 12) if x != 10], sep=r'\s+', header=0) # data from xfoil
    # df2 = pd.read_csv('naca2412polars.txt', skiprows=[x for x in range(0, 12) if x != 10], sep=r'\s+',
    #                  header=0)  # data from xfoil
    # df3 = pd.read_csv('naca2424polars.txt', skiprows=[x for x in range(0, 12) if x != 10], sep=r'\s+',
    #                   header=0)  # data from xfoil

    cl_experimental_data = pd.read_csv('clvsa.csv', sep = ',', names = ['alpha', 'Cl'])
    cm_experimental_data = pd.read_csv('cmvsa.csv', sep = ',', names = ['alpha', 'Cm'])
    cm_experimental_data.sort_values(by='alpha', inplace=True)


    # #plotting CL vs alpha
    plt.rc('font', size=15)
    plt.plot(df['alpha'], df['CL'], label='xfoil 2408')
    # plt.plot(df2['alpha'], df2['CL'], label='xfoil 2412')
    # plt.plot(df3['alpha'], df3['CL'], label='xfoil 2424')
    plt.plot(angles, Cl, label='thin airfoil')
    plt.plot(cl_experimental_data['alpha'], cl_experimental_data['Cl'], label='Experimental Data', linestyle='dashed')
    plt.legend()
    plt.xlabel(r'$\alpha$ [deg]')
    plt.ylabel(r'$C_l$')
    plt.show()
    #
    #plotting Cm vs alpha
    plt.plot(df['alpha'], df['CM'], label='xfoil')
    plt.plot(angles, Cm, label='thin airfoil')
    plt.plot(cm_experimental_data['alpha'], cm_experimental_data['Cm'], label='Experimental Data', linestyle='dashed')
    plt.legend()
    plt.xlabel(r'$\alpha$ [deg]')
    plt.ylabel(r'$C_m$')
    plt.show()

    question2a()
    question2b()




