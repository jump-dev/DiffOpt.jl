# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module DiffOpt

using LinearAlgebra
import LinearAlgebra: ⋅, dot, Diagonal
using SparseArrays
using JuMP

import LazyArrays
import BlockDiagonals

import MathOptInterface as MOI
const MOIU = MOI.Utilities

import MathOptSetDistances as MOSD

const VI = MOI.VariableIndex
const CI = MOI.ConstraintIndex

include("utils.jl")
include("product_of_sets.jl")
include("diff_opt.jl")
include("moi_wrapper.jl")
include("jump_moi_overloads.jl")

include("bridges.jl")

include("QuadraticProgram/QuadraticProgram.jl")
include("ConicProgram/ConicProgram.jl")

"""
    add_all_model_constructors(model)

Add all constructors of [`AbstractModel`](@ref) defined in this package to
`model` with [`add_model_constructor`](@ref).
"""
function add_all_model_constructors(model)
    add_model_constructor(model, QuadraticProgram.Model)
    add_model_constructor(model, ConicProgram.Model)
    return
end

export diff_optimizer

end # module
