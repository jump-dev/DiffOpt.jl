using DiffOpt
using MathOptInterface
using Test

const MOI = MathOptInterface;
const MOIU = MathOptInterface.Utilities;

@testset "gen_random_problem" begin
    include("gen_random_problem.jl")
end
