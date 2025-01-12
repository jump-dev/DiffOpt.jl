# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module DiffOpt

using JuMP

import BlockDiagonals
import LazyArrays
import LinearAlgebra
import MathOptInterface as MOI
import MathOptSetDistances as MOSD
import ParametricOptInterface as POI
import SparseArrays

include("utils.jl")
include("product_of_sets.jl")
include("diff_opt.jl")
include("moi_wrapper.jl")
include("jump_moi_overloads.jl")
include("parameters.jl")

include("copy_dual.jl")
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

# TODO
# add precompilation statements

end # module
