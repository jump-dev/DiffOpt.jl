# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

import Pkg
Pkg.add(;
    url = "https://github.com/jump-dev/ParametricOptInterface.jl",
    rev = "master",
)

using Test

@testset "$file" for file in readdir(@__DIR__)
    if !endswith(file, ".jl") || file in ("runtests.jl", "utils.jl")
        continue
    end
    include(file)
end
