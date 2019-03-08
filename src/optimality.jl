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
## =============================================================================