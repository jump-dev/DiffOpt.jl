# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# ============================================================================
# GeneralModel forward differentiation
# ============================================================================

function DiffOpt.forward_differentiate!(model::GeneralModel)
    model.diff_time = @elapsed begin
        _build_A!(model)
        _ensure_basis!(model)
        m = length(model.ci_list)

        # 1. Build db from input_cache (1-based row indexing)
        db = zeros(m)
        for (F, S) in keys(model.input_cache.scalar_constraints.dict)
            for (ci, func) in model.input_cache.scalar_constraints[F, S]
                if !isempty(func.terms)
                    error(
                        "BasisLinearProgram: constraint coefficient " *
                        "perturbation (dA) is not supported.",
                    )
                end
                row = model.ci_to_row[ci]
                db[row] = -MOI.constant(func)
            end
        end

        # 2. Solve B * dx_B = db
        dx_B = model.B_lu \ db

        # 3. Map basic structural variables back to MOI VIs
        model.forw_dx = Dict{MOI.VariableIndex,Float64}()
        for (k, col) in enumerate(model.basic_structural)
            model.forw_dx[model.vi_list[col]] = dx_B[k]
        end
    end
    return
end
