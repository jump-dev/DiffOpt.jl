module DiffOpt

using Random
using LinearAlgebra
using MathOptInterface

const MOI = MathOptInterface;
const MOIU = MathOptInterface.Utilities;

include("./gen_random_problem.jl")
include("./utils.jl")

export generate_lp, generate_qp

end # module
