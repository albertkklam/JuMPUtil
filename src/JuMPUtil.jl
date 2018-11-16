__precompile__()

module JuMPUtil
    ## utilities
    include("deriv.jl")
    export populate_hess_sparse
    export populate_hess_sparse!
    export f!
    export g!
    export h!
    export j! ## TODO
    export c! ## TODO
    export check_unconstr
    export check_constr  ## TODO
end # module
