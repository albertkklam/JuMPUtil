__precompile__()

module JuMPUtil
    ## utilities
    include("deriv.jl")
    export populate_hess_sparse
    export populate_hess_sparse!
    export f!
    export g!
    export H!
    export h! ## TODO: multiple constraints
    export j! ## TODO: multiple constraints
    export c! ## TODO: multiple constraints
    export check_unconstr
    export check_constr  ## TODO

    include("util.jl")
    export inertia
    export setup
    export getvalue  # extend
    export setvalue  # extend
end # module
