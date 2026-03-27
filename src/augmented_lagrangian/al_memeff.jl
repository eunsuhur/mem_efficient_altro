mutable struct MemEffConVal{C,V,M}
    con::C
    inds::Vector{Int}
    vals::Vector{V}
    λ::Vector{V}
    μ::Vector{V}
    c_max::Vector{Float64}
    params::ConstraintParams{Float64}
    jac::M
    tmp::M
    ∇proj::Matrix{Float64}
    ∇²proj::Matrix{Float64}
end

function MemEffConVal(n::Int, m::Int, con::TO.AbstractConstraint, inds::AbstractVector{Int})
    con isa TO.CoupledConstraint && throw(ArgumentError("memory_efficient=true does not support coupled constraints"))

    p = length(con)
    P = length(inds)
    vals = [zeros(p) for _ = 1:P]
    λ = [zeros(p) for _ = 1:P]
    μ = [ones(p) for _ = 1:P]
    jac = Matrix{Float64}(TO.gen_jacobian(con))
    tmp = similar(jac)
    params = ConstraintParams()
    MemEffConVal(
        con,
        collect(inds),
        vals,
        λ,
        μ,
        zeros(P),
        params,
        jac,
        tmp,
        zeros(p, p),
        zeros(p, p),
    )
end

mutable struct MemEffConstraintSet{T} <: TO.AbstractConstraintSet
    convals::Vector{MemEffConVal}
    c_max::Vector{T}
    μ_max::Vector{T}
    μ_maxes::Vector{Vector{T}}
    p::Vector{Int}
end

function MemEffConstraintSet(cons::TO.ConstraintList, model::RD.AbstractModel)
    ncon = length(cons)
    convals = [MemEffConVal(cons.n, cons.m, cons[i], cons.inds[i]) for i = 1:ncon]
    c_max = zeros(Float64, ncon)
    μ_max = zeros(Float64, ncon)
    μ_maxes = [zeros(Float64, length(ind)) for ind in cons.inds]
    MemEffConstraintSet{Float64}(convals, c_max, μ_max, μ_maxes, copy(cons.p))
end

TO.get_convals(conSet::MemEffConstraintSet) = conSet.convals
TO.get_errvals(conSet::MemEffConstraintSet) = conSet.convals
Base.length(conSet::MemEffConstraintSet) = length(conSet.convals)
Base.iterate(conSet::MemEffConstraintSet) = length(conSet) == 0 ? nothing : (conSet.convals[1].con, 1)
Base.iterate(conSet::MemEffConstraintSet, i::Int) = i < length(conSet) ? (conSet.convals[i + 1].con, i + 1) : nothing
TO.num_constraints(conSet::MemEffConstraintSet) = conSet.p
TO.sense(cval::MemEffConVal) = TO.sense(cval.con)
Base.length(cval::MemEffConVal) = length(cval.con)

get_duals(conSet::MemEffConstraintSet) = [cval.λ for cval in conSet.convals]
function set_duals!(conSet::MemEffConstraintSet, λ)
    for i = 1:length(conSet)
        set_duals!(conSet.convals[i], λ[i])
    end
end
function set_duals!(cval::MemEffConVal, λ)
    for i in eachindex(cval.λ)
        cval.λ[i] .= λ[i]
    end
end

function TO.evaluate!(cval::MemEffConVal, Z::RD.AbstractTrajectory)
    TO.evaluate!(cval.vals, cval.con, Z, cval.inds)
end

function TO.evaluate!(conSet::MemEffConstraintSet, Z::RD.AbstractTrajectory)
    for conval in conSet.convals
        TO.evaluate!(conval, Z)
    end
end

@inline _to_svec(v::AbstractVector) = SVector{length(v)}(v)

function TO.cost!(J::Vector{<:Real}, conSet::MemEffConstraintSet)
    for conval in conSet.convals
        sense = TO.sense(conval)
        for (i, k) in enumerate(conval.inds)
            c = _to_svec(conval.vals[i])
            λ = _to_svec(conval.λ[i])
            μ = _to_svec(conval.μ[i])
            J[k] += TO.cost(sense, λ, c, μ)
        end
    end
end

function TO.max_violation!(conSet::MemEffConstraintSet)
    for (j, conval) in enumerate(conSet.convals)
        sense = TO.sense(conval.con)
        for i in eachindex(conval.inds)
            conval.c_max[i] = TO.max_violation(sense, _to_svec(conval.vals[i]))
        end
        conSet.c_max[j] = isempty(conval.c_max) ? 0.0 : maximum(conval.c_max)
    end
    return nothing
end

function TO.max_violation(conSet::MemEffConstraintSet)
    TO.max_violation!(conSet)
    isempty(conSet.c_max) ? 0.0 : maximum(conSet.c_max)
end

function TO.findmax_violation(conSet::MemEffConstraintSet)
    TO.max_violation!(conSet)
    isempty(conSet.c_max) && return "No constraints violated"
    c_max0, j_con = findmax(conSet.c_max)
    c_max0 < eps() && return "No constraints violated"
    conval = conSet.convals[j_con]
    i_con = argmax(conval.c_max)
    k_con = conval.inds[i_con]
    con_sense = TO.sense(conval.con)
    viol = abs.(TO.violation(con_sense, conval.vals[i_con]))
    _, i_max = findmax(viol)
    con_name = string(typeof(conval.con).name)
    return con_name * " at time step $k_con at " * TO.con_label(conval.con, i_max)
end

function max_penalty!(conSet::MemEffConstraintSet)
    for (j, conval) in enumerate(conSet.convals)
        maxes = conSet.μ_maxes[j]
        for i in eachindex(conval.μ)
            maxes[i] = maximum(conval.μ[i])
        end
        conSet.μ_max[j] = isempty(maxes) ? 0.0 : maximum(maxes)
    end
    return nothing
end

function set_params!(cval::MemEffConVal, opts)
    p = cval.params
    p.use_default[1] && (p.ϕ = opts.penalty_scaling)
    p.use_default[2] && (p.μ0 = opts.penalty_initial)
    p.use_default[3] && (p.μ_max = opts.penalty_max)
    p.use_default[4] && (p.λ_max = opts.dual_max)
end

function reset_duals!(conSet::MemEffConstraintSet)
    for conval in conSet.convals, λ in conval.λ
        λ .= 0
    end
end

function reset_penalties!(conSet::MemEffConstraintSet)
    for conval in conSet.convals, μ in conval.μ
        μ .= conval.params.μ0
    end
end

function reset!(conSet::MemEffConstraintSet)
    reset_duals!(conSet)
    reset_penalties!(conSet)
end

function reset!(conSet::MemEffConstraintSet, opts::SolverOptions{T}) where T
    for conval in conSet.convals
        set_params!(conval, opts)
    end
    opts.reset_duals && reset_duals!(conSet)
    opts.reset_penalties && reset_penalties!(conSet)
end

function dual_update!(conSet::MemEffConstraintSet)
    for conval in conSet.convals
        λ_max = conval.params.λ_max
        sense = TO.sense(conval.con)
        for i in eachindex(conval.inds)
            λ = _to_svec(conval.λ[i])
            c = _to_svec(conval.vals[i])
            μ = _to_svec(conval.μ[i])
            conval.λ[i] .= dual_update(sense, λ, c, μ, λ_max)
        end
    end
end

function penalty_update!(conSet::MemEffConstraintSet)
    for conval in conSet.convals
        ϕ = conval.params.ϕ
        μ_max = conval.params.μ_max
        for μ in conval.μ
            μ .*= ϕ
            clamp!(μ, 0, μ_max)
        end
    end
end

struct MemEffALObjective{T,O<:TO.Objective,C<:MemEffConstraintSet} <: TO.AbstractObjective
    obj::O
    constraints::C
end

TO.get_J(obj::MemEffALObjective) = TO.get_J(obj.obj)
Base.length(obj::MemEffALObjective) = length(obj.obj)
RD.state_dim(obj::MemEffALObjective) = RD.state_dim(obj.obj)
RD.control_dim(obj::MemEffALObjective) = RD.control_dim(obj.obj)
TO.ExpansionCache(obj::MemEffALObjective) = TO.ExpansionCache(obj.obj)

function TO.cost!(obj::MemEffALObjective, Z::RD.AbstractTrajectory)
    TO.cost!(obj.obj, Z)
    TO.evaluate!(obj.constraints, Z)
    TO.cost!(TO.get_J(obj), obj.constraints)
end

function add_constraint_gradient_hessian!(E::TO.Expansion{n,m,T}, conval::MemEffConVal, i::Int, z::RD.AbstractKnotPoint) where {n,m,T}
    fill!(conval.jac, 0.0)
    TO.jacobian!(conval.jac, conval.con, z)
    cx, cu = TO.gen_views(conval.jac, conval.con, n, m)
    tx, tu = TO.gen_views(conval.tmp, conval.con, n, m)

    c = _to_svec(conval.vals[i])
    λ = _to_svec(conval.λ[i])
    μ = _to_svec(conval.μ[i])
    sense = TO.sense(conval.con)

    if sense == TO.Equality()
        λbar = λ + μ[1] .* c
        size(cx, 2) > 0 && mul!(E.q.data, Transpose(cx), λbar, 1.0, 1.0)
        size(cu, 2) > 0 && mul!(E.r.data, Transpose(cu), λbar, 1.0, 1.0)
        μs = μ[1]
        size(cx, 2) > 0 && mul!(E.Q.data, Transpose(cx), cx, μs, 1.0)
        size(cu, 2) > 0 && mul!(E.R.data, Transpose(cu), cu, μs, 1.0)
        size(cx, 2) > 0 && size(cu, 2) > 0 && mul!(E.H.data, Transpose(cu), cx, μs, 1.0)
    elseif sense == TO.Inequality()
        a = @. (c >= 0) | (λ > 0)
        λbar = λ + μ[1] .* (a .* c)
        copyto!(conval.tmp, conval.jac)
        @inbounds for r = 1:length(a)
            conval.tmp[r, :] .*= a[r]
        end
        size(cx, 2) > 0 && mul!(E.q.data, Transpose(cx), λbar, 1.0, 1.0)
        size(cu, 2) > 0 && mul!(E.r.data, Transpose(cu), λbar, 1.0, 1.0)
        μs = μ[1]
        size(tx, 2) > 0 && mul!(E.Q.data, Transpose(cx), tx, μs, 1.0)
        size(tu, 2) > 0 && mul!(E.R.data, Transpose(cu), tu, μs, 1.0)
        size(tx, 2) > 0 && size(cu, 2) > 0 && mul!(E.H.data, Transpose(cu), tx, μs, 1.0)
    elseif sense isa TO.SecondOrderCone
        λbar = λ - μ .* c
        λp = TO.projection(sense, λbar)
        TO.∇projection!(sense, conval.∇proj, λbar)
        TO.∇²projection!(sense, conval.∇²proj, λbar, λp)

        mul!(conval.tmp, conval.∇proj, conval.jac)
        size(tx, 2) > 0 && mul!(E.q.data, Transpose(tx), λp, -1.0, 1.0)
        size(tu, 2) > 0 && mul!(E.r.data, Transpose(tu), λp, -1.0, 1.0)
        μs = μ[1]
        size(tx, 2) > 0 && mul!(E.Q.data, Transpose(tx), tx, μs, 1.0)
        size(tu, 2) > 0 && mul!(E.R.data, Transpose(tu), tu, μs, 1.0)
        size(tx, 2) > 0 && size(tu, 2) > 0 && mul!(E.H.data, Transpose(tu), tx, μs, 1.0)

        mul!(conval.tmp, conval.∇²proj, conval.jac)
        size(tx, 2) > 0 && mul!(E.Q.data, Transpose(cx), tx, μs, 1.0)
        size(tu, 2) > 0 && mul!(E.R.data, Transpose(cu), tu, μs, 1.0)
        size(tx, 2) > 0 && size(cu, 2) > 0 && mul!(E.H.data, Transpose(cu), tx, μs, 1.0)
    else
        throw(ArgumentError("unsupported constraint sense in memory-efficient AL path"))
    end
    return E
end

function stage_constraint_expansion!(E::TO.Expansion{n,m,T}, conset::MemEffConstraintSet, Z::RD.Traj, k::Int) where {n,m,T}
    for conval in conset.convals
        i = stage_index(conval.inds, k)
        i == 0 && continue
        add_constraint_gradient_hessian!(E, conval, i, Z[k])
    end
    return E
end

function stage_cost_expansion!(E::TO.Expansion, obj::MemEffALObjective, Z::RD.Traj, k::Int, cache)
    stage_objective_expansion!(E, obj.obj, Z, k, cache)
    stage_constraint_expansion!(E, obj.constraints, Z, k)
end

struct MemEffAugmentedLagrangianSolver{T,S<:AbstractSolver,C<:MemEffConstraintSet} <: ConstrainedSolver{T}
    opts::SolverOptions{T}
    stats::SolverStats{T}
    solver_uncon::S
    constraints::C
end

Base.size(solver::MemEffAugmentedLagrangianSolver) = size(solver.solver_uncon)
TO.cost(solver::MemEffAugmentedLagrangianSolver) = TO.cost(solver.solver_uncon)
TO.get_trajectory(solver::MemEffAugmentedLagrangianSolver) = TO.get_trajectory(solver.solver_uncon)
TO.get_objective(solver::MemEffAugmentedLagrangianSolver) = TO.get_objective(solver.solver_uncon)
TO.get_model(solver::MemEffAugmentedLagrangianSolver) = TO.get_model(solver.solver_uncon)
get_initial_state(solver::MemEffAugmentedLagrangianSolver) = get_initial_state(solver.solver_uncon)
TO.integration(solver::MemEffAugmentedLagrangianSolver) = TO.integration(solver.solver_uncon)
TO.get_constraints(solver::MemEffAugmentedLagrangianSolver) = solver.constraints
solvername(::Type{<:MemEffAugmentedLagrangianSolver}) = :MemEffAugmentedLagrangian

function MemEffAugmentedLagrangianSolver(
        prob::Problem{Q,T},
        opts::SolverOptions=SolverOptions(),
        stats::SolverStats=SolverStats(parent=solvername(MemEffAugmentedLagrangianSolver));
        solver_uncon=iLQRSolver,
        kwarg_opts...
    ) where {Q,T}
    set_options!(opts; kwarg_opts...)
    opts.memory_efficient = true

    conset = MemEffConstraintSet(prob.constraints, prob.model)
    obj = MemEffALObjective{T,typeof(prob.obj),typeof(conset)}(prob.obj, conset)
    rollout!(prob)
    prob_al = Problem{Q}(prob.model, obj, ConstraintList(size(prob)...),
        prob.x0, prob.xf, prob.Z, prob.N, prob.t0, prob.tf)

    solver_uncon = solver_uncon(prob_al, opts, stats)
    solver = MemEffAugmentedLagrangianSolver(opts, stats, solver_uncon, conset)
    reset!(solver)
    return solver
end

function reset!(solver::MemEffAugmentedLagrangianSolver)
    reset_solver!(solver)
    reset!(solver.solver_uncon)
end

function initialize!(solver::MemEffAugmentedLagrangianSolver)
    set_verbosity!(solver)
    clear_cache!(solver)
    reset!(solver)
    TO.cost!(TO.get_objective(solver), TO.get_trajectory(solver))
end

function record_iteration!(solver::MemEffAugmentedLagrangianSolver, J::T, c_max::T) where T
    conSet = TO.get_constraints(solver)
    max_penalty!(conSet)
    penalty_max = isempty(conSet.μ_max) ? zero(T) : maximum(conSet.μ_max)
    record_iteration!(solver.stats, c_max=c_max, penalty_max=penalty_max, is_outer=true)
end

function set_tolerances!(solver::MemEffAugmentedLagrangianSolver{T},
        solver_uncon::AbstractSolver{T}, i::Int,
        cost_tol=solver.opts.cost_tolerance,
        grad_tol=solver.opts.gradient_tolerance
    ) where T
    if i != solver.opts.iterations_outer
        solver_uncon.opts.cost_tolerance = solver.opts.cost_tolerance_intermediate
        solver_uncon.opts.gradient_tolerance = solver.opts.gradient_tolerance_intermediate
    else
        solver_uncon.opts.cost_tolerance = cost_tol
        solver_uncon.opts.gradient_tolerance = grad_tol
    end
    return nothing
end

function evaluate_convergence(solver::MemEffAugmentedLagrangianSolver)
    i = solver.stats.iterations
    solver.stats.c_max[i] < solver.opts.constraint_tolerance ||
        solver.stats.penalty_max[i] >= solver.opts.penalty_max
end

dual_update!(solver::MemEffAugmentedLagrangianSolver) = dual_update!(TO.get_constraints(solver))
penalty_update!(solver::MemEffAugmentedLagrangianSolver) = penalty_update!(TO.get_constraints(solver))

function solve!(solver::MemEffAugmentedLagrangianSolver{T,S}) where {T,S}
    initialize!(solver)
    conSet = TO.get_constraints(solver)
    solver_uncon = solver.solver_uncon::S
    J_ = TO.get_J(TO.get_objective(solver))
    cost_tol = solver.opts.cost_tolerance
    grad_tol = solver.opts.gradient_tolerance

    for i = 1:solver.opts.iterations_outer
        set_tolerances!(solver, solver_uncon, i, cost_tol, grad_tol)
        solve!(solver.solver_uncon)
        status(solver) > SOLVE_SUCCEEDED && break

        J = sum(J_)
        TO.max_violation!(conSet)
        c_max = isempty(conSet.c_max) ? zero(T) : maximum(conSet.c_max)
        record_iteration!(solver, J, c_max)

        evaluate_convergence(solver) && break

        dual_update!(solver)
        penalty_update!(solver)
        set_verbosity!(solver)
        reset!(solver_uncon)

        if i == solver.opts.iterations_outer
            solver.stats.status = MAX_ITERATIONS
        end
    end

    solver.opts.cost_tolerance = cost_tol
    solver.opts.gradient_tolerance = grad_tol
    terminate!(solver)
    return solver
end
