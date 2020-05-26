# DiffOpt.jl
Differentiating convex optimization program (`JuMP` or `MOI` models) w.r.t. program parameters.


## TODO
- [x] Example scripts/problems in MOI form - LPs, QPs
- [x] Obtaining dual using `Dualization.jl`
- [x] Solve primal, dual problem. Test the solution gap
- [x] Write complementary slackness conditions. refer/use `BilevelJuMP.jl`
- [ ] Test the solution in complementarity constraints (**fails**)

## Note
- Package developed using [PkgTemplates](https://github.com/invenia/PkgTemplates.jl)
