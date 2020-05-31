using DiffOpt
using MathOptInterface
using Test

const MOI = MathOptInterface;
const MOIU = MathOptInterface.Utilities;

@testset "utils" begin
    include("utils.jl")
end
