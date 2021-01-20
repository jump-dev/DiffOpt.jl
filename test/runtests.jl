using DiffOpt
using MathOptInterface
using Test
using OSQP
using Ipopt
using Clp
using Random
using SCS
using LinearAlgebra
using DelimitedFiles
using GLPK

const MOI = MathOptInterface;
const MOIU = MathOptInterface.Utilities;
const MOIT = MathOptInterface.Test

const ATOL = 1e-4
const RTOL = 1e-4

@testset "Examples" begin
    include(joinpath(@__DIR__, "../examples/solve-LP.jl"))
    include(joinpath(@__DIR__, "../examples/solve-QP.jl"))
end

@testset "Generate random problems" begin
    include("gen_random_problem.jl")
end

@testset "MOI_wrapper" begin
    include("MOI_wrapper.jl")
end

@testset "Utility Methods" begin
    include("utils.jl")
end

@testset "Solver Interface" begin
    include("solver_interface.jl")
end
