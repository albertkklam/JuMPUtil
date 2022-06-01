__precompile__()

module JuMPUtil
## imports
using JuMP
using LinearAlgebra
using SparseArrays

import MathOptInterface
const MOI = MathOptInterface

## utilities
include("deriv.jl")
export populate_hess_sparse
export populate_hess_sparse!
export f!
export g!
export H!
export h!  ## TODO: multiple constraints
export j!  ## TODO: multiple constraints
export c!  ## TODO: multiple constraints
# export check_unconstr  ## TODO
# export check_constr  ## TODO

include("util.jl")
# export inertia ## TODO: use HSL and update for v1.1
export setup
end # module
