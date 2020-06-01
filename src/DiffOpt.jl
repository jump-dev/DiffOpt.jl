module DiffOpt

using Random
using MathOptInterface

const MOI = MathOptInterface;
const MOIU = MathOptInterface.Utilities;

include("utils.jl")

export generate_lp

end # module
