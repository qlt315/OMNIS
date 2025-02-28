import cvxpy as cp
print(cp.installed_solvers())
import cvxpy as cp

# Define the variable x as an integer variable
M = 10
x = cp.Variable(M, integer=True)  # Enforce x to be integer-valued
t_vector = [1, 18, 2, 1, 2,5, 7, 2, 6, 2]
e_vector = [5, 7, 2, 6, 2,5, 7, 2, 6, 2]

# Objective function: t_vector / x + e_vector / x
objective = cp.Minimize(cp.sum(t_vector @ cp.inv_pos(x) + e_vector @ cp.inv_pos(x)))

# Add constraints
constraints = [x >= 1, x <= 2042, cp.sum(x) <= 2042]  # x >= 1 to avoid division by zero

# Problem definition
problem = cp.Problem(objective, constraints)

# Solve the problem using a solver that supports integer programming (e.g., ECOS_BB, SCIP)
problem.solve(solver=cp.GUROBI)  # ECOS_BB is a mixed-integer solver supported by CVXPY

print("Optimal value of x:", x.value)