using Test
import DiffOpt
import Random
import SCS
import LinearAlgebra
import DelimitedFiles

import SparseArrays: spzeros

import MathOptInterface
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const MOIT = MathOptInterface.Test

const ATOL = 2e-4
const RTOL = 2e-4

@testset "MOI_wrapper" begin
    @testset "Utils" begin
        include("utils.jl")
    end
    @testset "MOI_wrapper main" begin
        include("moi_wrapper.jl")
    end
    @testset "QP fwd" begin
        include("qp_forward.jl")
    end
    @testset "Conic bck" begin
        include("conic_backward.jl")
    end
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

@testset "Examples" begin
    @testset "autotuning-ridge" begin
        include(joinpath(@__DIR__, "../docs/src/examples/autotuning-ridge.jl"))
    end
    @testset "chainrules_unit" begin
        include(joinpath(@__DIR__, "../docs/src/examples/chainrules_unit.jl"))
    end
    @testset "custom-relu" begin
        include(joinpath(@__DIR__, "../docs/src/examples/custom-relu.jl")) # needs downloads
    end
    @testset "matrix-inversion-manual" begin
       include(joinpath(@__DIR__, "../docs/src/examples/matrix-inversion-manual.jl"))
    end
    @testset "sensitivity-analysis-ridge" begin
        include(joinpath(@__DIR__, "../docs/src/examples/sensitivity-analysis-ridge.jl"))
    end
    @testset "sensitivity-analysis-svm" begin
        include(joinpath(@__DIR__, "../docs/src/examples/sensitivity-analysis-svm.jl"))
    end
end
