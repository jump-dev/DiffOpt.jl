# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Documenter
using DiffOpt
using Literate
using Test

const _EXAMPLE_DIR = joinpath(@__DIR__, "src", "examples")

"""
    _include_sandbox(filename)
Include the `filename` in a temporary module that acts as a sandbox. (Ensuring
no constants or functions leak into other files.)
"""
function _include_sandbox(filename)
    mod = @eval module $(gensym()) end
    return Base.include(mod, filename)
end

function _file_list(full_dir, relative_dir, extension)
    return map(
        file -> joinpath(relative_dir, file),
        filter(file -> endswith(file, extension), sort(readdir(full_dir))),
    )
end

function literate_directory(dir)
    rm.(_file_list(dir, dir, ".md"))
    for filename in _file_list(dir, dir, ".jl")
        # `include` the file to test it before `#src` lines are removed. It is
        # in a testset to isolate local variables between files.
        @testset "$(filename)" begin
            _include_sandbox(filename)
        end
        Literate.markdown(filename, dir; documenter = true)
    end
    return
end

literate_directory(_EXAMPLE_DIR)

makedocs(;
    modules = [DiffOpt],
    doctest = false,
    clean = true,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.MathJax2(),
    ),
    pages = [
        "Home" => "index.md",
        "Introduction" => "intro.md",
        "Manual" => "manual.md",
        "Usage" => "usage.md",
        "Reference" => "reference.md",
        "Tutorials" => [
            joinpath("examples", f) for
            f in readdir(_EXAMPLE_DIR) if endswith(f, ".md")
        ],
    ],
    strict = true,  # See https://github.com/JuliaOpt/JuMP.jl/issues/1576
    repo = "https://github.com/jump-dev/DiffOpt.jl",
    sitename = "DiffOpt.jl",
    authors = "JuMP Community",
)

deploydocs(repo = "github.com/jump-dev/DiffOpt.jl.git", push_preview = true)
