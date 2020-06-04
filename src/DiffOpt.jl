module DiffOpt

using Random
using MathOptInterface

const MOI = MathOptInterface;
const MOIU = MathOptInterface.Utilities;

include("./gen_random_problem.jl")

export generate_lp, generate_qp

end # module
