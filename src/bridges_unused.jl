# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

function MOI.set(
    model::MOI.ModelLike,
    attr::BackwardInVariablePrimal,
    bridge::MOI.Bridges.Variable.VectorizeBridge,
    value,
)
    return MOI.set(model, attr, bridge.variable, value)
end
