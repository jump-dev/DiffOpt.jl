using DiffOpt
using Test
using OSQP
using Ipopt
using Clp
using Random
using SCS
using LinearAlgebra
using DelimitedFiles
using GLPK

using SparseArrays: spzeros

import MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const MOIT = MathOptInterface.Test

const ATOL = 1e-4
const RTOL = 1e-4

@testset "Examples" begin
    include(joinpath(@__DIR__, "../docs/src/examples/autotuning-ridge.jl"))
    include(joinpath(@__DIR__, "../docs/src/examples/chainrules_unit.jl"))
    # include(joinpath(@__DIR__, "../docs/src/examples/custom-relu.jl")) # needs downloads
    include(joinpath(@__DIR__, "../docs/src/examples/matrix-inversion-manual.jl")) # rev
    # include(joinpath(@__DIR__, "../docs/src/examples/sensitivity-analysis-ridge.jl")) # bug
    include(joinpath(@__DIR__, "../docs/src/examples/sensitivity-analysis-svm.jl"))
    # @joaquimg to @matbesancon: tutorials or tests or remove?
    include(joinpath(@__DIR__, "../examples/unit_example.jl"))
    include(joinpath(@__DIR__, "../examples/chainrules.jl"))
end

@testset "Generate random problems" begin
    include("gen_random_problem.jl")
end

@testset "MOI_wrapper" begin
    include("utils.jl")
    include("moi_wrapper.jl")
    include("qp_forward.jl")
    include("conic_backward.jl")
end

@testset "JuMP wrapper" begin
    include("jump.jl")
end

@testset "Solver Interface" begin
    include("solver_interface.jl")
end

@testset "Singular error with deleted variables: Sensitivity index issue" begin
    include("singular_exception.jl")
end
