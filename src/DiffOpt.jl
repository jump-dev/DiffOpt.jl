module DiffOpt

using LinearAlgebra
import LinearAlgebra: ⋅, dot, Diagonal
using SparseArrays
using JuMP

import LazyArrays
import BlockDiagonals

import MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

using MathOptSetDistances
const MOSD = MathOptSetDistances

const VI = MOI.VariableIndex
const CI = MOI.ConstraintIndex

const SUPPORTED_OBJECTIVES = Union{
    MOI.VariableIndex,
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64}
}

const SUPPORTED_SCALAR_SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64}
}

const SUPPORTED_SCALAR_FUNCTIONS = Union{
    MOI.VariableIndex,
    MOI.ScalarAffineFunction{Float64}
}

const SUPPORTED_VECTOR_FUNCTIONS = Union{
    MOI.VectorOfVariables,
    MOI.VectorAffineFunction{Float64},
}

const SUPPORTED_VECTOR_SETS = Union{
    MOI.Zeros,
    MOI.Nonpositives,
    MOI.Nonnegatives,
    MOI.SecondOrderCone,
    MOI.PositiveSemidefiniteConeTriangle,
}

include("utils.jl")
include("product_of_sets.jl")
include("diff_opt.jl")
include("QuadraticProgram/QuadraticProgram.jl")
include("ConicProgram/ConicProgram.jl")
include("moi_wrapper.jl")
include("jump_moi_overloads.jl")

include("bridges.jl")

export diff_optimizer

end # module
