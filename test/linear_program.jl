# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestLinearProgram

using Test
import DiffOpt
import HiGHS
import MathOptInterface as MOI
import SCS

const ATOL = 1e-2
const RTOL = 1e-2

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$name", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

include(joinpath(@__DIR__, "utils.jl"))

function test_differentiating_LP_checking_gradients_for_non_active_contraints()
    # Issue #40 from Gurobi.jl
    # min  x
    # s.t. x >= 0
    #      x >= 3
    nz = 1
    qp_test_with_solutions(
        HiGHS.Optimizer;
        q = ones(nz),
        G = -ones(2, nz),
        h = [0.0, -3.0],
        dzb = ones(nz),
        dqf = ones(nz),
        # Expected solutions
        dGb = [0.0, 3.0],
        dhb = [0.0, -1.0],
    )
    return
end

function test_differentiating_a_simple_LP_with_GreaterThan_constraint()
    # this is canonically same as above test
    # min  x
    # s.t. x >= 3
    nz = 1
    qp_test_with_solutions(
        HiGHS.Optimizer;
        q = ones(nz),
        G = -ones(1, nz),
        h = [-3.0],
        dzb = ones(nz),
        dqf = ones(nz),
        # Expected solutions
        dGb = [3.0],
        dhb = [-1.0],
    )
    return
end

function test_differentiating_lp()
    # refered from - https://en.wikipedia.org/wiki/Simplex_algorithm#Example
    # max 2x + 3y + 4z
    # s.t. 3x+2y+z <= 10
    #      2x+5y+3z <= 15
    #      x,y,z >= 0
    nz = 3
    qp_test_with_solutions(
        SCS.Optimizer;
        q = [-2.0, -3.0, -4.0],
        G = [
            3.0 2.0 1.0
            2.0 5.0 3.0
            -1.0 0.0 0.0
            0.0 -1.0 0.0
            0.0 0.0 -1.0
        ],
        h = [10.0, 15.0, 0.0, 0.0, 0.0],
        dzb = ones(nz),
        dqf = ones(nz),
        # Expected solutions
        dqb = zeros(nz),
        dGb = [
            0.0 0.0 0.0
            0.0 0.0 -5/3
            0.0 0.0 5/3
            0.0 0.0 -10/3
            0.0 0.0 0.0
        ],
        dhb = [0.0, 1 / 3, -1 / 3, 2 / 3, 0.0],
    )
    return
end

function test_differentiating_LP_with_variable_bounds()
    # max 2x + 3y + 4z
    # s.t. 3x+2y+z <= 10
    #      2x+5y+3z <= 15
    #      x ≤ 3
    #      0 ≤ y ≤ 2
    #      z ≥ 2
    #      x,y,z >= 0
    # variant of previous test with same solution
    nz = 3
    qp_test_with_solutions(
        HiGHS.Optimizer;
        q = [-2.0, -3.0, -4.0],
        G = [
            3.0 2.0 1.0
            2.0 5.0 3.0
            -1.0 0.0 0.0
            0.0 -1.0 0.0
            0.0 0.0 -1.0
            1.0 0.0 0.0
            0.0 1.0 0.0
            0.0 0.0 1.0
        ],
        h = [10.0, 15.0, 0.0, 0.0, 0.0, 3.0, 2.0, 6.0],
        dzb = ones(nz),
        dqf = ones(nz),
        # Expected solutions
        dqb = zeros(nz),
        dGb = [
            0.0 0.0 0.0
            0.0 0.0 -5/3
            0.0 0.0 5/3
            0.0 0.0 -10/3
            0.0 0.0 0.0
            0.0 0.0 0.0
            0.0 0.0 0.0
            0.0 0.0 0.0
        ],
        dhb = [0.0, 1 / 3, -1 / 3, 2 / 3, 0.0, 0.0, 0.0, 0.0],
    )
    return
end

function test_differentiating_LP_with_variable_bounds_2()
    nz = 3
    qp_test_with_solutions(
        HiGHS.Optimizer;
        q = [-2.0, -3.0, -4.0],
        G = [
            3.0 2.0 1.0
            2.0 5.0 3.0
            0.0 -1.0 0.0
            0.0 0.0 -1.0
        ],
        h = [10.0, 15.0, 0.0, 0.0],
        fix_indices = [1],
        fix_values = [0.0],
        dzb = ones(nz),
        dqf = ones(nz),
        # Expected solutions
        dqb = zeros(nz),
        dGb = [
            0.0 0.0 0.0
            0.0 0.0 -5/3
            0.0 0.0 -10/3
            0.0 0.0 0.0
        ],
        dhb = [0.0, 1 / 3, 2 / 3, 0.0],
        dAb = [0.0 0.0 -5 / 3],
        dbb = [1 / 3],
    )
    return
end

"""
max  2x + 3y + 4z
s.t. 3x+2y+z <= 10
     2x+5y+3z <= 15
     x ≤ 3
     0 ≤ y ≤ 2
     6 ≥ z ≥ -1
     x, y, z >= 0
variant of previous test with same solution
"""
function test_differentiating_lp_with_saf_with_le_ge_constraints()
    nz = 3
    qp_test_with_solutions(
        HiGHS.Optimizer;
        q = [-2.0, -3.0, -4.0],
        G = [
            3.0 2.0 1.0
            2.0 5.0 3.0
            -1.0 0.0 0.0
            0.0 -1.0 0.0
            0.0 0.0 -1.0
        ],
        h = [10.0, 15.0, 0.0, 0.0, 0.0], #5
        ub_indices = [1, 2, 3],
        ub_values = [3.0, 2.0, 6.0],
        lb_indices = [1],
        lb_values = [-1.0],
        dzb = ones(nz),
        dqb = zeros(nz),
        dGb = [
            0.0 0.0 0.0
            0.0 0.0 -5/3
            0.0 0.0 5/3
            0.0 0.0 -10/3
            0.0 0.0 0.0
            0.0 0.0 0.0
            0.0 0.0 0.0
            0.0 0.0 0.0
            0.0 0.0 0.0
        ],
        dhb = [0.0, 1 / 3, -1 / 3, 2 / 3, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    return
end

function test_differentiating_lp_with_nonactive_constraints()
    # Issue #40 from Gurobi.jl
    # min  x
    # s.t. x >= 0
    #      x >= 3
    qp_test_with_solutions(
        HiGHS.Optimizer;
        q = [1.0],
        G = -ones(2, 1),
        h = [0.0, -3.0],
        dzb = -ones(1),
        dhf = [0.0, 1.0],
        # Expected solutions
        z = [3.0],
        λ = [0.0, 1.0],
        dzf = -ones(1),
        dλb = zeros(2),
        dhb = [0.0, 1.0],
        ∇zb = zeros(1),
        ∇λb = [0.0, -1.0],
        dλf = zeros(2),
    )
    return
end

end  # module

TestLinearProgram.runtests()
