module DiffOpt

using Random
using LinearAlgebra
using MathOptInterface
using BlockDiagonals
using SparseArrays
using IterativeSolvers

const MOI = MathOptInterface;
const MOIU = MathOptInterface.Utilities;

include("./gen_random_problem.jl")
include("./utils.jl")
include("./moi_wrapper_methods.jl")
include("./MOI_wrapper.jl")


export diff_optimizer, Optimizer, backward!, backward_conic!
export is_equality  # just for reference sake
export generate_lp, generate_qp

end # module
