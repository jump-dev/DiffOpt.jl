module DiffOpt

using Random
using LinearAlgebra
using SparseArrays
using IterativeSolvers: lsqr
using JuMP

import BlockDiagonals

using MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

using MathOptSetDistances
const MOSD = MathOptSetDistances

using MatrixOptInterface
const MatOI = MatrixOptInterface

include("gen_random_problem.jl")
include("moi_wrapper_methods.jl")
include("conic_diff.jl")
include("quadratic_diff.jl")
include("MOI_wrapper.jl")
include("utils.jl")


export diff_optimizer
export generate_lp, generate_qp

end # module
