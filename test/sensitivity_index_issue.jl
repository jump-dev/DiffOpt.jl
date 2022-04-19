using JuMP
import HiGHS

function DiffOptModel()
    model = DiffOpt.diff_optimizer(HiGHS.Optimizer)
    MOI.set(model, DiffOpt.ProgramClass(), DiffOpt.QUADRATIC)
    return model
end

#TEST OF DIFFOPT
@testset "Sensitivity index issue" begin
    testModel = Model(() -> DiffOptModel())
    #Variables
    @variable(testModel, v[i in 1:1])
    @variable(testModel, u[i in 1:1])
    @variable(testModel, g[j in 1:3])
    @variable(testModel, a)
    #Contraints
    @constraint(testModel, v_cons_min, v[1] >= 0.0)
    @constraint(testModel, u_cons_min, u[1] >= 0.0)
    @constraint(testModel, G1_cons_min, g[1] >= 0.0)
    @constraint(testModel, G2_cons_min, g[2] >= 0.0)
    @constraint(testModel, G3_cons_min, g[3] >= 0.0)
    @constraint(testModel, a_cons_min, a >= 0.0)

    @constraint(testModel, v_cons_max, v[1] <= 50.0)
    @constraint(testModel, G1_cons_max, g[1] <= 15.0)
    @constraint(testModel, G2_cons_max, g[2] <= 20.0)
    @constraint(testModel, G3_cons_max, g[3] <= 25.0)

    @constraint(testModel, future_constraint, v[1] + a >= 50)

    @constraint(testModel, hidro_conservation[i in 1:1], v[1] + u[i] == 45.0)
    @constraint(testModel, demand_constraint, g[1] + g[2] + g[3] + u[1] == 100)

    #Stage objective
    @objective(testModel, Min, 3.0*g[1]+5.0*g[2]+7.0*g[3]+a)

    #Calculation of sensitivities by Manual KKT

    #v,u,g1,g2,g3,a
    A = [
        1.0 1.0 0.0 0.0 0.0 0.0;
        0.0 1.0 1.0 1.0 1.0 0.0
    ]
    #rhs
    b = [
        45.0;
        100.0
    ]
    #v,u,g1,g2,g3,a
    G = [
        -1.0 0.0 0.0 0.0 0.0 0.0;
        0.0 -1.0 0.0 0.0 0.0 0.0;
        0.0 0.0 -1.0 0.0 0.0 0.0;
        0.0 0.0 0.0 -1.0 0.0 0.0;
        0.0 0.0 0.0 0.0 -1.0 0.0;
        0.0 0.0 0.0 0.0 0.0 -1.0;
        1.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 1.0 0.0 0.0 0.0;
        0.0 0.0 0.0 1.0 0.0 0.0;
        0.0 0.0 0.0 0.0 1.0 0.0;
        -1.0 0.0 0.0 0.0 0.0 -1.0
    ]
    h = [#rhs
        0.0;
        0.0;
        0.0;
        0.0;
        0.0;
        0.0;
        50.0;
        15.0;
        20.0;
        25.0;
        -50.0
    ]
    optimize!(testModel)
    lambda = [
        JuMP.dual.(testModel[:v_cons_min]);
        JuMP.dual.(testModel[:u_cons_min]);
        JuMP.dual.(testModel[:G1_cons_min]);
        JuMP.dual.(testModel[:G2_cons_min]);
        JuMP.dual.(testModel[:G3_cons_min]);
        JuMP.dual.(testModel[:a_cons_min]);
        JuMP.dual.(testModel[:v_cons_max]);
        JuMP.dual.(testModel[:G1_cons_max]);
        JuMP.dual.(testModel[:G2_cons_max]);
        JuMP.dual.(testModel[:G3_cons_max]);
        JuMP.dual.(testModel[:future_constraint])
    ]
    lambda = abs.(lambda)
    z = [
        JuMP.value.(testModel[:v])[1];
        JuMP.value.(testModel[:u])[1];
        JuMP.value.(testModel[:g])[1];
        JuMP.value.(testModel[:g])[2];
        JuMP.value.(testModel[:g])[3];
        JuMP.value.(testModel[:a])
    ]

    Q = zeros(6,6)
    D_lambda = diagm(lambda)
    D_Gz_h = diagm(G*z-h)

    KKT = [
        Q (G') (A');
        diagm(lambda)*G diagm(G*z-h) zeros(11,2);
        A zeros(2, 11) zeros(2,2)
    ]
    rhsKKT = [
        zeros(6,11) zeros(6,2);
        diagm(lambda) zeros(11,2);
        zeros(2,11) diagm(ones(2));
    ]

    derivativeKKT = hcat([DiffOpt.lsqr(KKT,rhsKKT[:,i]) for i in 1:size(rhsKKT)[2]]...)
    
    dprimal_dconsKKT = derivativeKKT[1:6,:]
    #Finished calculation of sensitivities by Manual KKT

    #Calculation of sensitivities by DiffOpt
    xRef = [
        testModel[:v_cons_min];
        testModel[:u_cons_min];
        testModel[:G1_cons_min];
        testModel[:G2_cons_min];
        testModel[:G3_cons_min];
        testModel[:a_cons_min];
        testModel[:v_cons_max];
        testModel[:G1_cons_max];
        testModel[:G2_cons_max];
        testModel[:G3_cons_max];
        testModel[:future_constraint];
        testModel[:hidro_conservation];
        testModel[:demand_constraint];
    ]
    yRef = [
        testModel[:v];
        testModel[:u];
        testModel[:g];
        testModel[:a]
    ]
    dprimal_dcons = Array{Float64, 2}(undef, length(yRef), length(xRef))
    for i in 1:length(xRef)
        constraint_equation = convert(MOI.ScalarAffineFunction{Float64}, 1.0)
        MOI.set(testModel, DiffOpt.ForwardInConstraint(), xRef[i], constraint_equation)
        DiffOpt.forward_differentiate!(testModel)
        dprimal_dcons[:,i] .= MOI.get.(testModel, DiffOpt.ForwardOutVariablePrimal(), yRef)
        constraint_equation = convert(MOI.ScalarAffineFunction{Float64}, 0.0)
        MOI.set(testModel, DiffOpt.ForwardInConstraint(), xRef[i], constraint_equation)
    end

    @testset "Sensitivities Result" begin
        #The result given by Manual KKT needs to invert sign in some values to match the constraints.
        #The result given by DiffOpt needs to invert sign to be in the right side of the equation.
        @test -dprimal_dcons[:,1] ≈ -dprimal_dconsKKT[:,1] atol=ATOL
        @test -dprimal_dcons[:,2] ≈ -dprimal_dconsKKT[:,2] atol=ATOL
        @test -dprimal_dcons[:,3] ≈ -dprimal_dconsKKT[:,3] atol=ATOL
        @test -dprimal_dcons[:,4] ≈ -dprimal_dconsKKT[:,4] atol=ATOL
        @test -dprimal_dcons[:,5] ≈ -dprimal_dconsKKT[:,5] atol=ATOL
        @test -dprimal_dcons[:,6] ≈ -dprimal_dconsKKT[:,6] atol=ATOL
        @test -dprimal_dcons[:,7] ≈ dprimal_dconsKKT[:,7] atol=ATOL
        @test -dprimal_dcons[:,8] ≈ dprimal_dconsKKT[:,8] atol=ATOL
        @test -dprimal_dcons[:,9] ≈ dprimal_dconsKKT[:,9] atol=ATOL
        @test -dprimal_dcons[:,10] ≈ dprimal_dconsKKT[:,10] atol=ATOL
        @test -dprimal_dcons[:,11] ≈ -dprimal_dconsKKT[:,11] atol=ATOL
        @test -dprimal_dcons[:,12] ≈ dprimal_dconsKKT[:,12] atol=ATOL
        @test -dprimal_dcons[:,13] ≈ dprimal_dconsKKT[:,13] atol=ATOL
    end
    @testset "Primal Results" begin
        @test JuMP.value.(testModel[:g]) == [15.0, 20.0, 20.0]
        @test JuMP.value.(testModel[:v]) == [0.0]
        @test JuMP.value.(testModel[:u]) == [45.0]
        @test JuMP.value(testModel[:a]) == 50.0
    end
end
