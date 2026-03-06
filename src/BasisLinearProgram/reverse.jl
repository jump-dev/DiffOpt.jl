# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# ============================================================================
# GeneralModel reverse differentiation
# ============================================================================

function DiffOpt.reverse_differentiate!(model::GeneralModel)
    model.diff_time = @elapsed begin
        _build_A!(model)
        _ensure_basis!(model)
        m = length(model.ci_list)

        # 1. Build dL/dx_B (length = m)
        #    First n_structural entries = dL/dx for basic structural vars
        #    Remaining entries = 0 (slack variables, not observed by user)
        dl_dx_B = zeros(m)
        for (vi, val) in model.input_cache.dx
            # should not happen so better to throw
            # if !haskey(model.vi_to_col, vi)
            #     continue
            # end
            col = model.vi_to_col[vi]
            for (k, bc) in enumerate(model.basic_structural)
                if bc == col
                    dl_dx_B[k] = val
                    break
                end
            end
        end

        # 2. y = B⁻ᵀ dL/dx_B
        y = model.B_lu' \ dl_dx_B

        # 3. Store dL/db per constraint: sign convention dL/db = -y
        model.back_db = Dict{MOI.ConstraintIndex,Float64}()
        for (i, ci) in enumerate(model.ci_list)
            model.back_db[ci] = -y[i]
        end
    end
    return
end
