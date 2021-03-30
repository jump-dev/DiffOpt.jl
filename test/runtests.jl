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

import MatrixOptInterface
const MatOI = MatrixOptInterface

const ATOL = 1e-4
const RTOL = 1e-4

@testset "Examples" begin
    include(joinpath(@__DIR__, "../examples/solve-LP.jl"))
    include(joinpath(@__DIR__, "../examples/solve-QP.jl"))
    include(joinpath(@__DIR__, "../examples/unit_example.jl"))
    include(joinpath(@__DIR__, "../examples/sensitivity-SVM.jl"))
end

@testset "Generate random problems" begin
    include("gen_random_problem.jl")
end

@testset "MOI_wrapper" begin
    include("MOI_wrapper.jl")
end

@testset "JuMP wrapper" begin
    include("jump.jl")
end

@testset "Solver Interface" begin
    include("solver_interface.jl")
end

@testset "Singular error with deleted variables" begin
    include("singular_exception.jl")
end
