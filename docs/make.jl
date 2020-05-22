using Documenter, DiffOpt

makedocs(;
    modules=[DiffOpt],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/AKS1996/DiffOpt.jl/blob/{commit}{path}#L{line}",
    sitename="DiffOpt.jl",
    authors="Akshay Sharma",
    assets=String[],
)
