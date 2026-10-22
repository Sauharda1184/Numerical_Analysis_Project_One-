from main import find_all_roots, horner

# Each entry is (coefficients, all_real): coefficients ascending by power,
# all_real True if every root should come back as a plain float (not complex).
# The polynomials cover real-only, complex-only, and mixed root cases.

TEST_CASES = [
    ([6, -5, 1], True),                 # (x-2)(x-3):              roots 2, 3
    ([-6, 11, -6, 1], True),            # (x-1)(x-2)(x-3):         roots 1, 2, 3
    ([4, 0, 5, 0, 1], False),           # (x^2+1)(x^2+4):          roots +/-i, +/-2i
    ([-24, 44, -30, 15, -6, 1], False), # (x-1)(x-2)(x-3)(x^2+4):  roots 1, 2, 3, +/-2i
    ([12, 2, -8, 2], True),             # 2x^3-8x^2+2x+12:         roots -1, 2, 3
    ([-4, -4, -4, 0, 1], False),        # x^4-4x^2-4x-4: Muller finds a real root here
]

RESIDUAL_TOL = 1e-4


def test_find_all_roots():
    all_passed = True

    for coefficients, all_real in TEST_CASES:
        degree = len(coefficients) - 1
        roots = find_all_roots(coefficients)

        # A degree-n polynomial must have exactly n roots (counted with
        # multiplicity); losing or duplicating one here would mean
        # synth_div deflated incorrectly.
        count_ok = len(roots) == degree

        # Each reported root must actually satisfy p(root) ~ 0.
        residuals = [abs(horner(coefficients, r)[0]) for r in roots]
        residuals_ok = all(res < RESIDUAL_TOL for res in residuals)

        # For all-real polynomials, roots should come back as plain
        # floats, not complex values with a zero imaginary part.
        type_ok = not all_real or not any(isinstance(r, complex) for r in roots)

        passed = count_ok and residuals_ok and type_ok
        all_passed = all_passed and passed

        status = "PASS" if passed else "FAIL"
        print(f"{status}: coefficients={coefficients}")
        print(f"      roots={roots}")
        print(f"      residuals={residuals}")

    return all_passed


if __name__ == "__main__":
    if test_find_all_roots():
        print("\nAll tests passed.")
    else:
        print("\nSome tests FAILED.")