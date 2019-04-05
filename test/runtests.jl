using JuMP, Ipopt
using SparseArrays, LinearAlgebra
include("../src/JuMPUtil.jl")
using ForwardDiff
using Test

## 1/2 xAx + bx
## -----------------------------------------------------------------------------
A = [10.0 1.0; 1.0 9.0]
b = [2.0; 3.0]
tol = 1e-9
m = Model(solver=IpoptSolver())
@variable(m, x[1:2])
@NLobjective(m, Min, 0.5 * sum(A[i, j] * x[i] * x[j] for i in eachindex(x)
                                                     for j in eachindex(x))
                                   + sum(b[i] * x[i] for i in eachindex(x)))

f = x -> 0.5 * (x'*A*x) + b'*x
g = x -> A*x + b
H = x -> A

d = JuMPUtil.setup(m)
xrand = randn(2)
grad = zeros(2)
hess_obj = spzeros(2, 2)

@testset "1/2 xAx + bx" begin
@test abs(JuMPUtil.f!(xrand, model=d) - f(xrand)) <= tol
@test (JuMPUtil.g!(grad, xrand, model=d); norm(grad - g(xrand)) <= tol)
@test (JuMPUtil.H!(hess_obj, xrand, model=d); norm(hess_obj - H(xrand)) <= tol)
@test (JuMPUtil.H!(hess_obj, xrand, model=d); JuMPUtil.H!(hess_obj, xrand, model=d); norm(hess_obj - H(xrand)) <= tol)
end

## 1/2 z(x)Az(x) + bz(x)
## -----------------------------------------------------------------------------
A = [10.0 1.0 0.0 ; 1.0 9.0 1.0; 0.0 1.0 8.0]
b = [2.0; 3.0; 4.0]
z = x -> cos.(x)
tol = 1e-9
m = Model(solver=IpoptSolver())
@variable(m, x[1:3])
@NLobjective(m, Min, 0.5 * sum(A[i, j] * cos(x[i]) * cos(x[j]) for i in eachindex(x)
                                                           for j in eachindex(x))
                                   + sum(b[i] * cos(x[i]) for i in eachindex(x)))

f = x -> 0.5 * (z(x)'*A*z(x)) + b'*z(x)
g = x -> ForwardDiff.gradient(f, x)
H = x -> ForwardDiff.hessian(f, x)

d = JuMPUtil.setup(m)
xrand = randn(3)
grad = zeros(3)
hess_obj = spzeros(3, 3)

@testset "1/2 z(x)Az(x) + bz(x)" begin
@test abs(JuMPUtil.f!(xrand, model=d) - f(xrand)) <= tol
@test (JuMPUtil.g!(grad, xrand, model=d); norm(grad - g(xrand)) <= tol)
@test (JuMPUtil.H!(hess_obj, xrand, model=d); norm(hess_obj - H(xrand)) <= tol)
@test (JuMPUtil.H!(hess_obj, xrand, model=d); JuMPUtil.H!(hess_obj, xrand, model=d); norm(hess_obj - H(xrand)) <= tol)
end

## Min  1/2 z(x)Az(x) + bz(x)
## s.t. ||z(x)||_2^2 = 0.5
## -----------------------------------------------------------------------------
A = [10.0 1.0 0.0 ; 1.0 9.0 1.0; 0.0 1.0 8.0]
b = [2.0; 3.0; 4.0]
z = x -> cos.(x)
tol = 1e-9
m = Model(solver=IpoptSolver())
@variable(m, x[1:3])
@NLobjective(m, Min, 0.5 * sum(A[i, j] * cos(x[i]) * cos(x[j]) for i in eachindex(x)
                                                           for j in eachindex(x))
                                   + sum(b[i] * cos(x[i]) for i in eachindex(x)))
@NLconstraint(m, constraint, sum(cos(x[i])^2 for i in eachindex(x)) == 0.5)

f = x -> 0.5 * (z(x)'*A*z(x)) + b'*z(x)
g = x -> ForwardDiff.gradient(f, x)
H = x -> ForwardDiff.hessian(f, x)
c = x -> norm(z(x))^2
cg = x -> ForwardDiff.gradient(c, x)
cH = x -> ForwardDiff.hessian(c, x)

d = JuMPUtil.setup(m)
xrand = randn(3)
grad = zeros(3)
jac = spzeros(1, 3)
hess_obj = spzeros(3, 3)
hess_con = spzeros(3, 3)

@testset "objective: 1/2 z(x)Az(x) + bz(x)" begin
@test abs(JuMPUtil.f!(xrand, model=d) - f(xrand)) <= tol
@test (JuMPUtil.g!(grad, xrand, model=d); norm(grad - g(xrand)) <= tol)
@test (JuMPUtil.H!(hess_obj, xrand, model=d); norm(hess_obj - H(xrand)) <= tol)
@test (JuMPUtil.H!(hess_obj, xrand, model=d); JuMPUtil.H!(hess_obj, xrand, model=d); norm(hess_obj - H(xrand)) <= tol)
end
@testset "constraint: ||z(x)||_2^2 = 0.5" begin
@test (JuMPUtil.j!(jac, xrand, model=d); norm(jac' - cg(xrand)) <= tol)
@test (JuMPUtil.h!(hess_con, xrand, model=d); norm(hess_con' - cH(xrand)) <= tol)
@test (JuMPUtil.h!(hess_con, xrand, model=d); JuMPUtil.h!(hess_con, xrand, model=d); norm(hess_con' - cH(xrand)) <= tol)
end

## Min  1/2 z(x)Az(x) + bz(x)
## s.t. ||z(x)||_2^2 = 0.5
##      x >= 0.0
##      x^2 >= 0.0
##      x^3 >= 0.0
## -----------------------------------------------------------------------------
A = [10.0 1.0 0.0 ; 1.0 9.0 1.0; 0.0 1.0 8.0]
b = [2.0; 3.0; 4.0]
z = x -> cos.(x)
tol = 1e-9
m = Model(solver=IpoptSolver())
@variable(m, x[1:3])
@NLobjective(m, Min, 0.5 * sum(A[i, j] * cos(x[i]) * cos(x[j]) for i in eachindex(x)
                                                           for j in eachindex(x))
                                   + sum(b[i] * cos(x[i]) for i in eachindex(x)))
@NLconstraint(m, constraint_a, sum(cos(x[i])^2 for i in eachindex(x)) == 0.5)
@constraint(m, constraint_b[i=1:3], x[i] >= 0.0)
@NLconstraint(m, constraint_c[i=1:3], x[i]^2 >= 0.0)
@NLconstraint(m, constraint_d[i=1:3], x[i]^3 >= 0.0)

f = x -> 0.5 * (z(x)'*A*z(x)) + b'*z(x)
g = x -> ForwardDiff.gradient(f, x)
H = x -> ForwardDiff.hessian(f, x)
ca = x -> norm(z(x))^2 - 0.5
cag = x -> ForwardDiff.gradient(ca, x)
caH = x -> ForwardDiff.hessian(ca, x)
cb = x -> x
cbg = x -> Diagonal([ForwardDiff.derivative(cb, x[i]) for i = 1:3])
cbH = x -> Diagonal([0.0 for i = 1:3])
cc = x -> x.^2
ccg = x -> Diagonal([ForwardDiff.derivative(cc, x[i]) for i = 1:3])
ccH = x -> Diagonal([2.0 for i = 1:3])
cd = x -> x.^3
cdg = x -> Diagonal([ForwardDiff.derivative(cd, x[i]) for i = 1:3])
cdH = x -> Diagonal(2x.^2)

d = JuMPUtil.setup(m)
xrand = randn(3)
grad = zeros(3)
jac = spzeros(1 + 3*3, 3)
constr = zeros(1 + 3*3)
hess_obj = spzeros(3, 3)
hess_con = spzeros(3, 3)
##            linear      nonlinear 1  nonlinear 2 nonlinear 3
Jac = Matrix([cbg(xrand); cag(xrand)'; ccg(xrand); cdg(xrand)])
C = [cb(xrand); ca(xrand); cc(xrand); cd(xrand)]

@testset "objective: 1/2 z(x)Az(x) + bz(x)" begin
@test abs(JuMPUtil.f!(xrand, model=d) - f(xrand)) <= tol
@test (JuMPUtil.g!(grad, xrand, model=d); norm(grad - g(xrand)) <= tol)
@test (JuMPUtil.H!(hess_obj, xrand, model=d); norm(hess_obj - H(xrand)) <= tol)
@test (JuMPUtil.H!(hess_obj, xrand, model=d); JuMPUtil.H!(hess_obj, xrand, model=d); norm(hess_obj - H(xrand)) <= tol)
end
@testset "constraint: ||z(x)||_2^2 = 0.5" begin
@test (JuMPUtil.j!(jac, xrand, model=d); JuMPUtil.j!(jac, xrand, model=d); norm(jac - Jac) <= tol)
@test (JuMPUtil.c!(constr, xrand, model=d); JuMPUtil.c!(constr, xrand, model=d); norm(constr - C) <= tol)
end

## 1/2 xAx + bx (sparse)
## -----------------------------------------------------------------------------
A = sparse([10.0 1.0 0.0; 1.0 9.0 2.0; 0.0 2.0 9.0])
b = [1.0; 2.0; 3.0]
tol = 1e-9
m = Model(solver=IpoptSolver())
@variable(m, x[1:3])
rows, cols, vals = findnz(A)
@NLobjective(m, Min, 0.5 * sum(A[r, c] * x[r] * x[c] for (r,c) in collect(zip(rows,cols)))
                                       + sum(b[i] * x[i] for i in eachindex(x)))

f = x -> 0.5 * (x'*A*x) + b'*x
g = x -> A*x + b
H = x -> A

d = JuMPUtil.setup(m)
xrand = randn(3)
grad = zeros(3)
hess_obj = spzeros(3, 3)

@testset "1/2 xAx + bx sparse" begin
@test abs(JuMPUtil.f!(xrand, model=d) - f(xrand)) <= tol
@test (JuMPUtil.g!(grad, xrand, model=d); norm(grad - g(xrand)) <= tol)
@test (JuMPUtil.H!(hess_obj, xrand, model=d); norm(hess_obj - H(xrand)) <= tol)
@test (JuMPUtil.H!(hess_obj, xrand, model=d); JuMPUtil.H!(hess_obj, xrand, model=d); norm(hess_obj - H(xrand)) <= tol)
end

## 1/2 xAx + bx (sparse 2)
## -----------------------------------------------------------------------------
A = sparse([10.0 1.0 1.5; 1.0 9.0 2.0; 1.5 2.0 9.0])
b = [1.0; 2.0; 3.0]
tol = 1e-9
m = Model(solver=IpoptSolver())
@variable(m, x[1:3])
rows, cols, vals = findnz(A)
xAx = @NLexpression(m, xAx, sum(A[r, c] * x[r] * x[c] for (r,c) in collect(zip(rows,cols))))
@NLobjective(m, Min, 0.5 * xAx + sum(b[i] * x[i] for i in eachindex(x)))

f = x -> 0.5 * (x'*A*x) + b'*x
g = x -> A*x + b
H = x -> A

d = JuMPUtil.setup(m)
xrand = randn(3)
grad = zeros(3)
hess_obj = spzeros(3, 3)

@testset "1/2 xAx + bx sparse" begin
@test abs(JuMPUtil.f!(xrand, model=d) - f(xrand)) <= tol
@test (JuMPUtil.g!(grad, xrand, model=d); norm(grad - g(xrand)) <= tol)
@test (JuMPUtil.H!(hess_obj, xrand, model=d); norm(hess_obj - H(xrand)) <= tol)
@test (JuMPUtil.H!(hess_obj, xrand, model=d); JuMPUtil.H!(hess_obj, xrand, model=d); norm(hess_obj - H(xrand)) <= tol)
end