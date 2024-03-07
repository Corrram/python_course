import sympy as sp
from sympy.parsing.latex import parse_latex

latex_str = r"\left(v\frac{(\theta*u*(v-1)-\theta*(u-1)(v-1)+1)}{(\theta*(u-1)(v-1)-1)^2}\right)^2"
integrand = parse_latex(latex_str)
integrand = sp.simplify(integrand)
symbols = [*integrand.free_symbols]
u = [sym for sym in symbols if sym.name == "u"][0]
v = [sym for sym in symbols if sym.name == "v"][0]
integral = sp.integrate(integrand, (u, 0, 1))
integral = sp.integrate(integrand, (u, 0, 1), (v, 0, 1))
print(6 * integral - 2)
print("Done!")
