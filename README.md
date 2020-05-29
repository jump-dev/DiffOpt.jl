# DiffOpt.jl
[![Build Status](https://travis-ci.org/AKS1996/DiffOpt.jl.svg?branch=master)](https://travis-ci.org/AKS1996/DiffOpt.jl) 
[![Coverage Status](https://coveralls.io/repos/github/AKS1996/DiffOpt.jl/badge.svg?branch=master)](https://coveralls.io/github/AKS1996/DiffOpt.jl?branch=master)


Differentiating convex optimization program (`JuMP` or `MOI` models) w.r.t. program parameters.


## TODO
- [x] Example scripts/problems in MOI form - LPs, QPs
- [x] Obtaining dual using `Dualization.jl`
- [x] Solve primal, dual problem. Test the solution gap
- [x] Write complementary slackness conditions. refer/use `BilevelJuMP.jl`
- [ ] Test the solution in complementarity constraints (**fails**)

## Note
- Package developed using [PkgTemplates](https://github.com/invenia/PkgTemplates.jl)
