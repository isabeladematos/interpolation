import numpy
import scipy.linalg
import matplotlib.pyplot as plt

def cspline( x, y, x_interp, B0 , Bk ):

    # h_i, v_i, d_i, i = 0, 1, ..., k - 1:
    h = x[ 1: ] - x[ :-1 ]
    v = y[ 1: ] - y[ :-1 ]
    d = v / h

    # 1 / h_i, i = 0, 1, ..., k - 1:
    off_diag = 1.0 / h
    # 2 * ( 1 / h_i + 1 / h_{i + 1} ), i = 0, 1, ..., k - 2
    diag = 2.0 * ( off_diag[ 1: ] + off_diag[ :-1 ] )
    # 1 / h_i, i = 1, 2, ..., k - 2:
    off_diag = off_diag[ 1:-1 ]

    # Compact tridiagonal hermitian matrix:
    A = numpy.zeros( ( 2, diag.size ) )
    A[ 0, 1: ] = off_diag[ : ]
    A[ 1, : ] = diag

    # Right-hand side:
    z = d / h
    b = 3.0 * ( z[ :-1 ] + z[ 1: ] )
    b[ 0 ] = b[ 0 ] - B0 / h[ 0 ]
    b[ -1 ] = b[ -1 ] - Bk / h[ -1 ]

    # B-coefficients:
    B = numpy.zeros( ( b.size + 2, ) )
    B[ 1:-1 ] = scipy.linalg.solve_banded( A, b )
    B[ 0 ] = B0
    B[ -1 ] = Bk

    # C-coefficients:
    C = ( 3.0 * d - 2.0 * B[ :-1 ] - B[ 1: ] ) / h
    # D-coefficients:
    D = ( B[ :-1 ] + B[ 1: ] - 2.0 * d ) / ( h ** 2 )

    # Interpolate:
    y_interp = numpy.zeros( x_interp.shape )
    k = 0
    for i in range( x_interp.size ):
        # Find appropriate segment:
        while k < C.size and x_interp[ i ] > x[ k + 1 ]:
            k = k + 1
        # Compute cubic polinomial:
        h_x = x_interp[ i ] - x[ k ]
        y_interp[ i ] = y[ k ] + B[ k ] * h_x + C[ k ] * ( h_x ** 2 ) + D[ k ] * ( h_x ** 3 )

    return y_interp

def runge( x ):

    return 1.0 / ( 1.0 + 25.0 * x**2 )


# 1-) splines naturais: s'(x0) = B0 = 0 e s'(xk) = Bk = 0

B0 = 0
Bk = 0

ek = []
for k in range(1,15):
    x_interp = numpy.linspace(-1.1,1.1,100)
   
    x = numpy.zeros(k+1)
    for i in range(k+1):
        x[i] = -1 + 2*i/k  

    sk = cspline( x, runge(x), x_interp, B0 , Bk )
    plt.plot(x_interp, runge(x_interp), 'r')
    plt.plot(x_interp, sk, 'b')
    plt.plot(x, runge(x), 'ko')
    plt.show()

    ek.append(numpy.amax(numpy.absolute(numpy.subtract(runge(x_interp), sk))))

