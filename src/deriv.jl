## =============================================================================
## get derivatives from JuMP
## =============================================================================
"""
...
## `populate_hess_sparse`: populates a sparse hessian from indices and values
### arguments:
- `i::Array{Int64,1}`: row indices
- `j::Array{Int64,1}`: col indices
- `h::Array{Int64,1}`: values s.t. h[1] = H[i[1], j[1]]
- `n::Int64`: dimension of hessian
### returns:
- `H::SparseMatrixCSC`: sparse, symmetric hessian
...
"""
function populate_hess_sparse(i::Array{Int64,1},
                              j::Array{Int64,1},
                              h::Array{Float64,1},
                              n::Int64)
    H = spzeros(n, n)
    @simd for idx in eachindex(i)
        ii = i[idx]; jj = j[idx]; hh = h[idx]
        ## !! NOTE: UNSAFE ACCORDING TO DOCS; NEED TO FILTER (i,j) and (j,i) !!
        if i[idx] != j[idx]
           @inbounds @views H[ii, jj] += hh
           @inbounds @views H[jj, ii] += hh
        else
           @inbounds @views H[ii, jj] += hh
        end
    end
    return H
end

"""
...
## `populate_hess_sparse!`: populates a sparse hessian from indices and values in-place or calls `populate_hess_sparse` based on `structure` flag
### arguments:
- `H::SparseMatrixCSC`: sparse, symmetric hessian
- `i::Array{Int64,1}`: row indices
- `j::Array{Int64,1}`: col indices
- `h::Array{Int64,1}`: values s.t. h[1] = H[i[1], j[1]]
- `structure::Int64`:
    - `structure` = `0`: indicates that `H` is empty outside of indices defined by `i,j` and of appropriate dimension; method will overwrite the `i,j` element with zero and then populate with `h`. This is needed to address dupes.
    - `structure` ≠ `0`: creates a sparse matrix of size `structure` × `structure`
...
"""
function populate_hess_sparse!(H::SparseMatrixCSC{Float64,Int64},
                               i::Array{Int64,1},
                               j::Array{Int64,1},
                               h::Array{Float64,1},
                               structure::Int64=0)
    @assert(length(i) == length(j)); @assert(length(i) == length(h))
    ## address structure
    if structure != 0
        ## `H` is "not fresh" outside of `i,j` elements; zero out `i,j` elements to be added to
        return populate_hess_sparse(i, j, h, structure)
    else
        ## `H` is "fresh" outside of `i,j` elements; zero out `i,j` elements to be added to
        @simd for idx in eachindex(i)
            ii = i[idx]; jj = j[idx]
            ## !! NOTE: UNSAFE ACCORDING TO DOCS; NEED TO FILTER (i,j) and (j,i) !!
            if i[idx] != j[idx]
               @inbounds @views H[ii, jj] = 0.0
               @inbounds @views H[jj, ii] = 0.0
            else
               @inbounds @views H[ii, ii] = 0.0
            end
        end
        ## populate elements
        @simd for idx in eachindex(i)
            ii = i[idx]; jj = j[idx]; hh = h[idx]
            if i[idx] != j[idx]
               @inbounds @views H[ii, jj] += hh
               @inbounds @views H[jj, ii] += hh
            else
               @inbounds @views H[ii, ii] += hh
            end
        end
    end
end

"""
...
## `populate_jac_sparse!`: populates a sparse jacobian from indices and values in-place
### arguments:
- `J::SparseMatrixCSC`: sparse jacobian
- `i::Array{Int64,1}`: row indices
- `j::Array{Int64,1}`: col indices
- `h::Array{Int64,1}`: values s.t. h[1] = H[i[1], j[1]]
- `n::Int64`: dimension of hessian
...
"""
function populate_jac_sparse!(J::SparseMatrixCSC{Float64,Int64},
                              i::Array{Int64,1}, j::Array{Int64,1},
                              jj::Array{Float64,1})
    for elem in zip(i, j, jj)
        J[elem[1], elem[2]] = elem[3]
    end
end

"""
...
## `f!`: evalue NLP model objective function
### arguments:
    * `x::Array{Float64,1}`: point to evaluate
    * `[model::JuMP.NLPEvaluator]`: (kwarg) initialized model to evaluate
...
"""
function f!(x::Array{Float64,1}; model::JuMP.NLPEvaluator)
    return MathProgBase.eval_f(model, x)
end

"""
...
## `g!`: evaluate gradient of model objective function in-place
### arguments:
    * `g::Array{Float64,1}`: placeholder for gradient
    * `x::Array{Float64,1}`: point to evaluate
    * `[model::JuMP.NLPEvaluator]`: (kwarg) initialized model to evaluate
...
"""
function g!(g::Array{Float64,1}, x::Array{Float64,1}; model::JuMP.NLPEvaluator)
    MathProgBase.eval_grad_f(model, g, x)
end

"""
...
## `H!`: evaluate Hessian of model objective function in-place
### arguments:
    * `H::SparseMatrixCSC{Float64,Int64}`: placeholder for hessian
    * `x::Array{Float64,1}`: point to evaluate
    * `[model::JuMP.NLPEvaluator]`: (kwarg) initialized model to evaluate
...
"""
function H!(H::SparseMatrixCSC{Float64,Int64}, x::Array{Float64,1};
            model::JuMP.NLPEvaluator, structure::Int64=0)
    m = MathProgBase.numconstr(model.m)
    ij_hess = MathProgBase.hesslag_structure(model)
    h = ones(length(ij_hess[1]))
    MathProgBase.eval_hesslag(model, h, x, 1.0, zeros(m))
    populate_hess_sparse!(H, ij_hess[1], ij_hess[2], h, structure)
end

## TODO: sum of hessians of constraints; ie multiple constraints...
"""
...
## `h!`: evaluate hessian of model constraint function(s??) in-place
### arguments:
    * `H::SparseMatrixCSC{Float64,Int64}`: placeholder for hessian
    * `x::Array{Float64,1}`: point to evaluate
    * `[model::JuMP.NLPEvaluator]`: (kwarg) initialized model to evaluate
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
## `j!`: evaluate jacobian of model constraint function(s??) in-place
### arguments:
    * `J::SparseMatrixCSC{Float64,Int64}`: placeholder for Jacobian
    * `x::Array{Float64,1}`: point to evaluate
    * `[model::JuMP.NLPEvaluator]`: (kwarg) initialized model to evaluate
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
"""
...
## `c!`: evaluate constraint(s??) of model in-place
### arguments:
    * `c::Array{Float64,1}`: placeholder for constraint evaluations
    * `x::Array{Float64,1}`: point to evaluate
    * `[model::JuMP.NLPEvaluator]`: (kwarg) initialized model to evaluate
...
"""
function c!(c::Array{Float64,1}, x::Array{Float64,1}; model::JuMP.NLPEvaluator)
    # m = MathProgBase.numconstr(model.m)
    # con = fill(NaN, m)
    MathProgBase.eval_g(model, c, x)
end
## =============================================================================