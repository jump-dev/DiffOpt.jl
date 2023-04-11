# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

import HiGHS
@testset "Differentiating LP; checking gradients for non-active contraints" begin
    # Issue #40 from Gurobi.jl
    # min  x
    # s.t. x >= 0
    #      x >= 3
    qp_test_with_solutions(
        HiGHS.Optimizer,
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
end
