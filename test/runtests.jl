using DiffOpt
using MathOptInterface
using Test
using OSQP
using GLPK

const MOI = MathOptInterface;
const MOIU = MathOptInterface.Utilities;

@testset "Generate random problems" begin
    include("gen_random_problem.jl")
end

@testset "diff_model" begin
    include("diff_model.jl")
end

@testset "Utility Methods" begin
    include("utils.jl")
end
