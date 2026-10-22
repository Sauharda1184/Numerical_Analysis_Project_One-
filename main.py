
import math
import cmath

def g(coefficients, x):
    # One Newton-Raphson step: x - p(x)/p'(x).
    value, derivative = horner(coefficients, x)

    return x - value / derivative

def newton(coefficients, p_0):
    # Newton's method: repeatedly apply g() from the real guess p_0 until
    # consecutive iterates are within TOL, or give up after 100 tries.
    TOL = 1e-6
    i = 1 

    while i <= 100:
        try:
            p = g(coefficients, p_0)
        except ZeroDivisionError:
            # p'(p_0) == 0, so the Newton step is undefined here (e.g. the
            # default guess p_0=0 on p(x)=x^2+1). Nudge off this point and
            # try again instead of crashing.
            p_0 += 1
            i += 1
            continue

        if (abs(p - p_0) < TOL):
            # Report f(p), the residual at the converged point, not another
            # Newton step (that was the old bug here: f(E)=g(coefficients, p)
            # computed one more x - p/p' instead of evaluating p(p)).
            print(f"Newton: I reached x={p} in {i} iterations. f(x)={horner(coefficients, p)[0]}")
            return p
        else:
            i += 1
            p_0 = p
    print("Newton's method failed. Did not reach TOL in 100 iterations.")
    # None (instead of just printing and falling off the end) lets callers
    # like find_all_roots() detect failure and fall back to muller().
    return None

def muller(coefficients, p0, p1, p2):
    '''Fits a parabola through (p0,f(p0)), (p1,f(p1)), (p2,f(p2)) and takes
     the parabola's root closest to p2 as the next estimate, then slides
     the window forward. Using cmath (not math) lets the discriminant go
     negative, which is what lets this converge to complex roots even
     when p0, p1, p2 all start out real.'''
    TOL = 1e-6
    f = lambda x: horner(coefficients, x)[0]

    # h1, h2: step sizes between the three points.
    # delta1, delta2: slopes of the secants p0->p1 and p1->p2.
    # d: the parabola's leading (quadratic) coefficient, from the second
    # divided difference of the three points.
    h1 = p1 - p0
    h2 = p2 - p1
    delta1 = (f(p1) - f(p0)) / h1
    delta2 = (f(p2) - f(p1)) / h2
    d = (delta2 - delta1) / (h2 + h1)

    i = 3
    while i <= 100:
        # Parabola through the 3 points, in the local variable h = x - p2:
        #   f(p2) + b*h + d*h^2 = 0
        # b is the parabola's slope at p2; D is its discriminant's root.
        b = delta2 + h2 * d
        D = cmath.sqrt(b**2 - 4 * f(p2) * d)

        # Of the quadratic formula's two forms (b+D)/2d and (b-D)/2d,
        # pick whichever denominator has larger magnitude. That denominator
        # is E here; it avoids catastrophic cancellation when b and D are
        # close in value.
        if abs(b - D) < abs(b + D):
            E = b + D
        else:
            E = b - D

        # h is the step from p2 to the new estimate p (this is the
        # quadratic formula solved for the root nearest p2).
        try:
            h = -2 * f(p2) / E
            p = p2 + h
        except ZeroDivisionError:
            # There is a low, but nonzero, chance that E = 0. For example, initial
            # guesses of p0, p1, p2 = 0, 1, 2 will have a zero division error for 
            # coefficients [2,-6,11,-6,1], or x^4 - 6x^3 + 11x^2 - 6x + 2. If there
            # is such an error, h becomes greater than the tolerance (arbitrarily
            #  chosen as 100 here) and p becomes a nudge of p0.
            h = 100
            p = p0 + 0.1
        

        if abs(h) < TOL:
            print(f"Muller: I reached x={p} in {i} iterations. f(p)={f(p)}")
            return p

        # Slide the window forward: drop p0, keep p1/p2, add the new estimate.
        p0, p1, p2 = p1, p2, p
        h1 = p1 - p0
        h2 = p2 - p1
        delta1 = (f(p1) - f(p0)) / h1
        delta2 = (f(p2) - f(p1)) / h2
        d = (delta2 - delta1) / (h2 + h1)
        i += 1

    print("Muller's method failed. Did not reach TOL in 100 iterations.")

def horner(coefficients, x):
    # Horner's method: evaluates p(x) and p'(x) together in one pass of
    # nested multiplication. y ends up as p(x); z ends up as p'(x).
    n = len(coefficients)-1
    y = coefficients[n]
    z = coefficients[n]
    for j in range (n-1, 0 ,-1):
        y = x * y + coefficients[j]
        z = x * z + y
    y = x * y + coefficients[0]
    return y, z

def synth_div(coefficients, r):
    '''Divides p(x) by (x - r) using synthetic division, so once a root r is
    found we can strip it out and keep hunting on a lower-degree polynomial
    (this is the "deflation" step of the deflation algorithm).

    Recurrence (b = quotient coefficients, a = coefficients, both ascending
    by power): b[n-1] = a[n]; b[k] = a[k+1] + r*b[k+1] for k = n-2..0.
    That recurrence is run as a descending loop below because each b[k]
    depends on the just-computed b[k+1].'''
    n = len(coefficients) - 1
    quotient = [0] * n
    quotient[n - 1] = coefficients[n]
    for k in range(n - 2, -1, -1):
        quotient[k] = coefficients[k + 1] + r * quotient[k + 1]
    # remainder = p(r); ~0 confirms r really is a root of the original p.
    remainder = coefficients[0] + r * quotient[0]
    return quotient, remainder

def solve_linear(coefficients):
    # coefficients = [a0, a1] for a0 + a1*x = 0  =>  x = -a0/a1.
    a0, a1 = coefficients
    return [-a0 / a1]

def solve_quadratic(coefficients):
    # coefficients = [a0, a1, a2] for a0 + a1*x + a2*x^2 = 0.
    a0, a1, a2 = coefficients
    disc = a1**2 - 4 * a2 * a0
    # cmath.sqrt always returns a complex, even for a non-negative input
    # (e.g. sqrt(4) -> (2+0j)), which is why real roots were printing as
    # things like (3-0j). Only reach for cmath when the root is actually
    # going to be complex: disc is already complex-typed (coefficients
    # went complex from an earlier deflation step) or disc < 0.
    if isinstance(disc, complex) or disc < 0:
        sqrt_disc = cmath.sqrt(disc)
    else:
        sqrt_disc = math.sqrt(disc)
    return [(-a1 + sqrt_disc) / (2 * a2), (-a1 - sqrt_disc) / (2 * a2)]

def find_all_roots(coefficients):
    '''The deflation driver: repeatedly finds one root and deflates until
    the polynomial is down to degree <=2, then solves that base case
    directly. This is the piece that ties newton/muller/horner/synth_div
    together into "find all roots of a degree <=5 polynomial."

    Root-finding strategy per deflation step: try newton() first (fast,
    but only ever finds real roots). If it fails to converge (returns
    None), fall back to muller(), which can land on complex roots even
    from real starting guesses via its complex sqrt.'''
    coefficients = list(coefficients)
    roots = []
    conjugate_root = None

    while len(coefficients) - 1 > 2:
        root = newton(coefficients, 0)
        if root is None:
            root = muller(coefficients, 0, 1, 2)
            if not (root is None):
                conjugate_root = root.conjugate()
                # Since we assume that our polynomial has real coefficients,
                # all of the imaginary roots will appear in conjugate pairs.
                # Rather than running Muller's method once for each complex
                # root, we can run it for each pair of complex roots.
        if root is None:
            raise RuntimeError(f"Could not find a root of {coefficients}; try different starting guesses.")

        roots.append(root)
        if not (conjugate_root is None):
            coefficients, _ = synth_div(coefficients, conjugate_root)
            roots.append(conjugate_root)
            conjugate_root = None
        coefficients, _ = synth_div(coefficients, root)

    # Base case: solve the last line (degree 2) or point (degree 1) root(s)
    # directly instead of iterating further.
    degree = len(coefficients) - 1
    if degree == 2:
        roots.extend(solve_quadratic(coefficients))
    elif degree == 1:
        roots.extend(solve_linear(coefficients))

    return roots

if __name__ == "__main__":
    a = [2,-6,11,-6,1]
    newton(a, 3)
    print(find_all_roots(a))