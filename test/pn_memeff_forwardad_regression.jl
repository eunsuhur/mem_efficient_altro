using ForwardDiff

@testset "Memory-Efficient PN ForwardAD Regression" begin
    struct DiscreteForwardADModel <: RD.AbstractModel end
    Base.size(::DiscreteForwardADModel) = (2, 1)
    RD.state_dim(::DiscreteForwardADModel) = 2
    RD.control_dim(::DiscreteForwardADModel) = 1
    RD.diffmethod(::DiscreteForwardADModel) = RD.ForwardAD()

    abstract type DiscreteForwardADExp <: RD.Explicit end

    function RD.discrete_dynamics(::Type{DiscreteForwardADExp}, ::DiscreteForwardADModel,
                                  x::AbstractVector, u::AbstractVector, t::Real, dt::Real)
        T = promote_type(eltype(x), eltype(u), typeof(dt))
        return T[
            x[1] + dt * x[2] + (dt^2 / 2) * u[1],
            x[2] + dt * u[1],
        ]
    end

    function RD._discrete_jacobian!(::RD.ForwardAD, ::Type{Q}, grad, model::DiscreteForwardADModel,
                                    z::RD.AbstractKnotPoint{T,N,M}) where {T,N,M,Q<:RD.Explicit}
        ix, iu, dt = z._x, z._u, z.dt
        t = z.t
        fd_aug(s) = RD.discrete_dynamics(Q, model, s[ix], s[iu], t, dt)
        grad .= ForwardDiff.jacobian(fd_aug, z.z)
        return nothing
    end

    function RD._discrete_jacobian!(::RD.ForwardAD, ::Type{Q}, grad, model::DiscreteForwardADModel,
                                    z::RD.AbstractKnotPoint{T,N,M}, cache) where {T,N,M,Q<:RD.Explicit}
        ix, iu, dt = z._x, z._u, z.dt
        t = z.t
        fd_aug(s) = RD.discrete_dynamics(Q, model, s[ix], s[iu], t, dt)
        grad .= ForwardDiff.jacobian(fd_aug, z.z)
        return nothing
    end

    function build_problem()
        model = DiscreteForwardADModel()
        n, m = size(model)
        N = 21
        tf = 2.0
        dt = tf / (N - 1)
        x0 = @SVector [0.0, 0.0]
        xf = @SVector [1.0, 0.0]

        Q = Diagonal(@SVector [1e-2, 1e-2])
        R = Diagonal(@SVector [1e-3])
        Qf = Diagonal(@SVector [10.0, 10.0])
        obj = LQRObjective(Q, R, Qf, xf, N, uf=@SVector [0.0])

        cons = ConstraintList(n, m, N)
        bnd = BoundConstraint(n, m, u_min=SVector(-1.2), u_max=SVector(1.2))
        add_constraint!(cons, bnd, 1:N-1)
        add_constraint!(cons, GoalConstraint(xf), N:N)

        X0 = [zeros(n) for _ = 1:N]
        U0 = [zeros(m) for _ = 1:N-1]
        X0[1] .= x0
        ts = collect(range(0.0, stop=tf, length=N))
        for k = 1:N-1
            X0[k + 1] .= RD.discrete_dynamics(DiscreteForwardADExp, model, X0[k], U0[k], ts[k], dt)
        end

        return Problem(model, obj, xf, tf,
            constraints=cons, t0=0.0, x0=x0, N=N, X0=X0, U0=U0, dt=dt,
            integration=DiscreteForwardADExp)
    end

    prob_al = build_problem()
    opts_al = SolverOptions(
        show_summary=false, verbose=0, verbose_pn=false,
        projected_newton=false, memory_efficient=true,
        constraint_tolerance=1e-6, projected_newton_tolerance=1e-2,
        iterations=50, iterations_inner=50, iterations_outer=5, n_steps=25,
    )
    solver_al = Altro.AugmentedLagrangianSolver(prob_al, opts_al)
    solve!(solver_al)
    c_al = TO.max_violation(solver_al)

    prob_pn = build_problem()
    opts_pn = SolverOptions(
        show_summary=false, verbose=0, verbose_pn=false,
        projected_newton=true, force_pn=true, memory_efficient=true,
        constraint_tolerance=1e-6, projected_newton_tolerance=1e-2,
        iterations=50, iterations_inner=50, iterations_outer=5, n_steps=25,
    )
    solver_pn = ALTROSolver(prob_pn, opts_pn)
    solve!(solver_pn)

    @test solver_pn.stats.iterations_pn > 0
    @test status(solver_pn) == Altro.SOLVE_SUCCEEDED
    @test TO.max_violation(solver_pn) < 1e-6
    @test TO.max_violation(solver_pn) <= c_al
end
