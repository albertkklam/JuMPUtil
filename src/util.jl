## =============================================================================
## helper functions
## =============================================================================
"""
## `setup`: initialize `JuMP` model `m` for derivative queries as an `NLPEvaluator`
"""
function setup(m::Model)
    d = JuMP.NLPEvaluator(m)
    MOI.initialize(d, [:Grad, :Jac, :Hess, :ExprGraph])
    return d
end

"""
...
## `inertia`: compute inertia (using bunch-kaufman LDLt)
### arguments:
    - `A::Array{Float64,2}`: (dense) symmetric matrix
### returns:
    - `Array{Float64,1}`: 3-array containing
        - number positive eig vals
        - number negative eig vals
        - number zero eig vals
...
"""
function inertia(A::Array{Float64,2})
    S = bunchkaufman(A)
    s = diag(S.D)
    n_pos = 0
    n_neg = 0
    n_zer = 0
    for e in s
        if e > 0
            n_pos += 1
        elseif e < 0
            n_neg += 1
        elseif e == 0
            n_zer += 1
        end
    end
    return [n_pos; n_neg; n_zer]
end
## =============================================================================
