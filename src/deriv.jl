using JuMP
using MathProgBase
using LinearAlgebra
using SparseArrays

## =============================================================================
## get derivatives from JuMP
## =============================================================================
"""
...
## `populate_hess_sparse(!)`: populates a sparse hessian from indices and values
### arguments:
- `i::Array{Int64,1}`: row indices
- `j::Array{Int64,1}`: col indices
- `h::Array{Int64,1}`: values s.t. h[1] = H[i[1], j[1]]
- `n::Int64`: dimension of hessian
### returns:
- `H::SparseMatrixCSC`: sparse, symmetric hessian
...
"""
function populate_hess_sparse(i::Array{Int64,1}, j::Array{Int64,1},
                              h::Array{Float64,1}, n::Int64)
    ## build matrix; probably a better way to do this...
    H = sparse([i;j], [j;i], [h;h], n, n, +)  ## NOTE: symmetrizing doubles the diagonal
    H = H - Diagonal(H)/2.                    ## NOTE: required removal of 1/2 digonal
    return H
end
function populate_hess_sparse!(H::SparseMatrixCSC{Float64,Int64},
                               i::Array{Int64,1}, j::Array{Int64,1},
                               h::Array{Float64,1})
    for elem in zip(i, j, h)
       if elem[1] != elem[2]
           H[elem[1], elem[2]] += elem[3]
           H[elem[2], elem[1]] += elem[3]
       else
           H[elem[1], elem[2]] += elem[3]
       end
    end
end

function populate_jac_sparse!(J::SparseMatrixCSC{Float64,Int64},
                              i::Array{Int64,1}, j::Array{Int64,1},
                              jj::Array{Float64,1})
    for elem in zip(i, j, jj)
        J[elem[1], elem[2]] = elem[3]
    end
end
"""
...
## `f!`: evalue model objective function
### arguments:
    * `x::Array{Float64,1}`: point to evaluate
    * `model::JuMP.NLPEvaluator`: initialized model to evaluate
...
"""
function f!(x::Array{Float64,1}; model::JuMP.NLPEvaluator)
    return MathProgBase.eval_f(model, x)
end

"""
...
## `g!`: evaluate gradient of model objective function in-place
### arguments:
    * `x::Array{Float64,1}`: point to evaluate
    * `g::Array{Float64,1}`: placeholder for gradient
    * `model::JuMP.NLPEvaluator`: initialized model to evaluate
...
"""
function g!(g::Array{Float64,1}, x::Array{Float64,1}; model::JuMP.NLPEvaluator)
    MathProgBase.eval_grad_f(model, g, x)
end

"""
...
## `H!`: evaluate Hessian of model objective function in-place
### arguments:
    * `x::Array{Float64,1}`: point to evaluate
    * `H::SparseMatrixCSC{Float64,Int64}`: placeholder for hessian
    * `model::JuMP.NLPEvaluator`: initialized model to evaluate
...
"""
function H!(H::SparseMatrixCSC{Float64,Int64}, x::Array{Float64,1}; model::JuMP.NLPEvaluator,
            lam::Array{Float64,1}=[0.0])
    ij_hess = MathProgBase.hesslag_structure(model)
    h = ones(length(ij_hess[1]))
    MathProgBase.eval_hesslag(model, h, x, 1.0, lam)
    H = populate_hess_sparse!(H, ij_hess[1], ij_hess[2], h)
end

## TODO: sum of hessians of constraints; ie multiple constraints...
"""
...
## `h!`: evaluate Hessian of model constraint function(s??) in-place
### arguments:
    * `x::Array{Float64,1}`: point to evaluate
    * `H::SparseMatrixCSC{Float64,Int64}`: placeholder for hessian
    * `model::JuMP.NLPEvaluator`: initialized model to evaluate
...
"""
function h!(H::SparseMatrixCSC{Float64,Int64}, x::Array{Float64,1}; model::JuMP.NLPEvaluator,
            lam::Array{Float64,1}=[1.0])
    ij_hess = MathProgBase.hesslag_structure(model)
    h = ones(length(ij_hess[1]))
    MathProgBase.eval_hesslag(model, h, x, 0.0, lam)
    H = populate_hess_sparse!(H, ij_hess[1], ij_hess[2], h)
end

## TODO: multiple constraints...
"""
...
## `j!`: evaluate Jacobian of model constraint function(s??) in-place
### arguments:
    * `x::Array{Float64,1}`: point to evaluate
    * `J::SparseMatrixCSC{Float64,Int64}`: placeholder for Jacobian
    * `model::JuMP.NLPEvaluator`: initialized model to evaluate
...
"""
function j!(J::SparseMatrixCSC{Float64,Int64}, x::Array{Float64,1}; model::JuMP.NLPEvaluator)
    m = MathProgBase.numconstr(model.m)
    n = MathProgBase.numvar(model.m)
    @assert(size(J) == (m,n))
    ij_jac = MathProgBase.jac_structure(model)
    j = ones(length(ij_jac[1]))
    MathProgBase.eval_jac_g(model, j, x)
    populate_jac_sparse!(J, ij_jac[1], ij_jac[2], j)
end
## TODO: multiple constraints
function c!(c::Array{Float64,1}, x::Array{Float64,1}; model::JuMP.NLPEvaluator)
    # m = MathProgBase.numconstr(model.m)
    # con = fill(NaN, m)
    MathProgBase.eval_g(model, c, x)
end


## =============================================================================
## check optimality conditions
## =============================================================================
"""
...
## `check`: checks fist and second order conditions for `model`
## reference: https://www.mathworks.com/help/optim/ug/constrained-nonlinear-optimization-algorithms.html
              http://www.cs.nthu.edu.tw/~cherung/teaching/2011cs5321/handout7.pdf
### arguments:
- `x::Array{Float64,1}`: point to evaluate
- `m::JuMP.NLPEvaluator`: initialized model to evaluate
...
"""
## TODO
function check_eqconstr(x::Array{Float64,1}, model::JuMP.NLPEvaluator)
    ## setup
    n = MathProgBase.numvar(model.m)
    m = MathProgBase.numconstr(model.m)

    ## get constraint names and multipliers
    keys = model.m.objDict.keys
    idx = [isassigned(keys, i) for i = 1:length(keys)]
    keys = keys[idx]
    non_vars = filter(x -> x == false, keys[(Symbol(e) in Symbol.(model.m.colNames)) for e in keys])
    var_idx = [(Symbol(e) in Symbol.(model.m.colNames)) for e in keys]
    con_idx = .~var_idx
    cons = keys[con_idx]
    lams = [getdual(model.m[cons[i]]) for i = 1:length(cons)]
    A = cons[lams != 0.0]

    ## first order necessary
    lam_star = getdual(model[:line_limit])
    obj, con, grad, hess, jac, KKT = query_model(model, x, 2; obj_mult=1.0, con_mult=[lam_star], kkt=true)
    grad = zeros()

    fon = grad - lam_star*vec(full(jac'))
    complementarity = lam_star .* con

    ## second order

    hess = spzeros()

    return norm(fon), complementarity
end

function check_unconstr(x::Array{Float64,1}, model::JuMP.NLPEvaluator; tol=1e-6)
    ## setup
    n = MathProgBase.numvar(model.m)
    m = MathProgBase.numconstr(model.m)

    ## first order necessary
    g = zeros(n)
    g!(g, x, model=model)
    norm_g = norm(g)
    @assert(norm_g <= tol)

    ## second order sufficent
    H = spzeros(n, n)
    h!(H, x, model=model)
    eigs = eigvals(Matrix(H))
    @assert(minimum(eigs) > 0.0)

    return norm_g, eigs
end
