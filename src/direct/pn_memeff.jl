"""
Memory-efficient Projected Newton solver using Riccati recursion.

Instead of forming full sparse constraint Jacobian (D) and Hessian (H) matrices,
exploits the block-tridiagonal KKT structure via a backward Riccati sweep
and forward rollout. Memory is O(N·(n+m)²) vs O(NP·NN) for the
standard dense Schur complement approach.

Based on: Rao, Wright, Rawlings (1998) "Application of Interior-Point
Methods to Model Predictive Control", J. Opt. Theory Appl.
"""
struct MemEffProjectedNewtonSolver{T,I<:QuadratureRule,N,M,NM,L<:AbstractModel,C} <: ConstrainedSolver{T}
    # Problem data
    model::L
    obj::Objective
    constraints::MemEffConstraintSet{T}
    x0::SVector{N,T}
    xf::SVector{N,T}

    # Trajectories
    Z::Traj{N,M,T,KnotPoint{T,N,M,NM}}
    Z̄::Traj{N,M,T,KnotPoint{T,N,M,NM}}

    # Solver state
    opts::SolverOptions{T}
    stats::SolverStats{T}
    Nsteps::Int

    # Backward pass gains (stored for forward pass)
    K::Vector{Matrix{T}}       # K[k] ∈ ℝ^{m×n}, k=1:N-1
    l::Vector{Vector{T}}       # l[k] ∈ ℝ^m, k=1:N-1

    # Dynamics workspace
    jac::Matrix{T}              # n×(n+m) for Jacobian computation

    # Riccati cost-to-go (alternating pair)
    Π::Vector{Matrix{T}}        # [curr, next], each n×n
    π_v::Vector{Vector{T}}      # [curr, next], each n

    # Q-function workspace
    Qxx::Matrix{T}              # n×n
    Qux::Matrix{T}              # m×n
    Quu::Matrix{T}              # m×m
    Qx::Vector{T}               # n
    Qu::Vector{T}               # m

    # General workspace
    tmp_nn::Matrix{T}           # n×n
    tmp_mn::Matrix{T}           # m×n
    tmp_nm::Matrix{T}           # n×m
    tmp_n1::Vector{T}           # n
    tmp_n2::Vector{T}           # n
    tmp_m::Vector{T}            # m

    # Caches
    fd_cache::C
    exp_cache::Any
end

function MemEffProjectedNewtonSolver(
        prob::Problem{QUAD,T},
        opts::SolverOptions=SolverOptions(),
        stats::SolverStats=SolverStats(parent=solvername(MemEffProjectedNewtonSolver));
        constraints::MemEffConstraintSet{T}=MemEffConstraintSet(prob.constraints, prob.model)
    ) where {QUAD,T}
    n,m,N = size(prob)

    Z = prob.Z
    Z̄ = copy(prob.Z)

    # Gains for each non-terminal stage
    K = [zeros(T, m, n) for _ = 1:N-1]
    l = [zeros(T, m) for _ = 1:N-1]

    # Dynamics Jacobian scratch
    jac = zeros(T, n, n+m)

    # Riccati matrices (pair for swapping)
    Π = [zeros(T, n, n), zeros(T, n, n)]
    π_v = [zeros(T, n), zeros(T, n)]

    # Q-function workspace
    Qxx = zeros(T, n, n)
    Qux = zeros(T, m, n)
    Quu = zeros(T, m, m)
    Qx = zeros(T, n)
    Qu = zeros(T, m)

    # General workspace
    tmp_nn = zeros(T, n, n)
    tmp_mn = zeros(T, m, n)
    tmp_nm = zeros(T, n, m)
    tmp_n1 = zeros(T, n)
    tmp_n2 = zeros(T, n)
    tmp_m = zeros(T, m)

    # Caches
    diffmethod = RD.diffmethod(prob.model)
    fd_cache = if diffmethod isa RD.ForwardAD
        nothing
    else
        FiniteDiff.JacobianCache(prob.model)
    end
    exp_cache = TO.ExpansionCache(prob.obj)

    MemEffProjectedNewtonSolver{T,QUAD,n,m,n+m,typeof(prob.model),typeof(fd_cache)}(
        prob.model, prob.obj, constraints,
        SVector{n}(prob.x0), SVector{n}(prob.xf),
        Z, Z̄, opts, stats, N,
        K, l, jac,
        Π, π_v,
        Qxx, Qux, Quu, Qx, Qu,
        tmp_nn, tmp_mn, tmp_nm, tmp_n1, tmp_n2, tmp_m,
        fd_cache, exp_cache
    )
end

# Interface
Base.size(solver::MemEffProjectedNewtonSolver{T,I,N,M}) where {T,I,N,M} = N, M, solver.Nsteps
TO.get_model(solver::MemEffProjectedNewtonSolver) = solver.model
TO.get_constraints(solver::MemEffProjectedNewtonSolver) = solver.constraints
TO.get_trajectory(solver::MemEffProjectedNewtonSolver) = solver.Z
TO.get_objective(solver::MemEffProjectedNewtonSolver) = solver.obj
iterations(solver::MemEffProjectedNewtonSolver) = solver.stats.iterations_pn
solvername(::Type{<:MemEffProjectedNewtonSolver}) = :MemEffProjectedNewton
TO.integration(::MemEffProjectedNewtonSolver{<:Any,Q}) where Q = Q

get_duals(solver::MemEffProjectedNewtonSolver) = get_duals(solver.constraints)
set_duals!(solver::MemEffProjectedNewtonSolver, λ) = set_duals!(solver.constraints, λ)

# ——————————————————————————————————————————————————————————
#  Solve
# ——————————————————————————————————————————————————————————
function solve!(solver::MemEffProjectedNewtonSolver)
    # Evaluate constraints at current trajectory
    TO.evaluate!(solver.constraints, solver.Z)
    TO.max_violation!(solver.constraints)

    if solver.opts.verbose_pn
        println("\nMemEff Projection:")
    end
    viol = projection_solve!(solver)
    if viol <= solver.opts.constraint_tolerance
        solver.stats.status = SOLVE_SUCCEEDED
    else
        solver.stats.status = MAX_ITERATIONS
    end
    terminate!(solver)
    return solver
end

function projection_solve!(solver::MemEffProjectedNewtonSolver)
    ϵ_feas = solver.opts.constraint_tolerance
    max_iters = solver.opts.n_steps

    # Compute initial violation
    TO.evaluate!(solver.constraints, solver.Z)
    TO.max_violation!(solver.constraints)
    viol = _memeff_max_violation(solver)

    count = 0
    while count <= max_iters && viol > ϵ_feas
        viol = _memeff_projection_step!(solver)
        count += 1
        _memeff_record_iteration!(solver, viol)
    end
    return viol
end

function _memeff_record_iteration!(solver::MemEffProjectedNewtonSolver, viol)
    J = TO.cost(solver.obj, solver.Z)
    record_iteration!(solver.stats, cost=J, c_max=viol, is_pn=true,
        dJ=0.0, gradient=NaN, penalty_max=NaN)
end

function _memeff_max_violation(solver::MemEffProjectedNewtonSolver)
    conSet = solver.constraints
    # Stage constraint violation
    c_max = isempty(conSet.c_max) ? 0.0 : maximum(conSet.c_max)
    # Dynamics violation
    d_max = _max_dynamics_violation(solver)
    return max(c_max, d_max)
end

function _max_dynamics_violation(solver::MemEffProjectedNewtonSolver{T,QUAD}) where {T,QUAD}
    Z = solver.Z
    N = solver.Nsteps
    d_max = zero(T)
    for k = 1:N-1
        x_next = RD.discrete_dynamics(QUAD, solver.model, Z[k])
        res = state(Z[k+1]) - x_next
        d_max = max(d_max, norm(res, Inf))
    end
    return d_max
end

# ——————————————————————————————————————————————————————————
#  Single projection step: backward Riccati + forward rollout
# ——————————————————————————————————————————————————————————
function _memeff_projection_step!(solver::MemEffProjectedNewtonSolver{T,QUAD,n,m}) where {T,QUAD,n,m}
    Z = solver.Z
    Z̄ = solver.Z̄
    N = solver.Nsteps
    conSet = solver.constraints
    ρ_primal = solver.opts.ρ_primal
    ρ_chol = solver.opts.ρ_chol

    # — Evaluate constraints at current trajectory —
    TO.evaluate!(conSet, Z)
    TO.max_violation!(conSet)

    # — Backward pass: compute gains K[k], l[k] via Riccati —
    _memeff_backward_pass!(solver)

    # — Forward pass with line search —
    viol0 = _memeff_max_violation(solver)
    if solver.opts.verbose_pn
        println("  feas0: $viol0")
    end

    α = 1.0
    ϕ = 0.5
    viol = viol0
    accepted = false
    for iter = 1:10
        _memeff_forward_rollout!(solver, α)

        TO.evaluate!(conSet, Z̄)
        TO.max_violation!(conSet)
        viol = _memeff_max_violation_candidate(solver)

        if solver.opts.verbose_pn
            println("  feas: $viol (α=$α)")
        end

        if viol < viol0
            # Accept step: copy Z̄ → Z
            for k in eachindex(Z)
                Z[k].z = Z̄[k].z
            end
            TO.evaluate!(conSet, Z)
            TO.max_violation!(conSet)
            accepted = true
            break
        end
        α *= ϕ
    end

    if !accepted
        TO.evaluate!(conSet, Z)
        TO.max_violation!(conSet)
        return viol0
    end

    return viol
end

function _memeff_max_violation_candidate(solver::MemEffProjectedNewtonSolver{T,QUAD}) where {T,QUAD}
    conSet = solver.constraints
    c_max = isempty(conSet.c_max) ? 0.0 : maximum(conSet.c_max)
    # Dynamics violation of candidate Z̄
    Z̄ = solver.Z̄
    N = solver.Nsteps
    d_max = zero(T)
    for k = 1:N-1
        x_next = RD.discrete_dynamics(QUAD, solver.model, Z̄[k])
        res = state(Z̄[k+1]) - x_next
        d_max = max(d_max, norm(res, Inf))
    end
    return max(c_max, d_max)
end

# ——————————————————————————————————————————————————————————
#  Backward pass: Riccati recursion with active constraints
# ——————————————————————————————————————————————————————————
function _memeff_backward_pass!(solver::MemEffProjectedNewtonSolver{T,QUAD,n,m}) where {T,QUAD,n,m}
    Z = solver.Z
    N = solver.Nsteps
    model = solver.model
    obj = solver.obj
    conSet = solver.constraints
    ρ_primal = solver.opts.ρ_primal
    ρ_chol = solver.opts.ρ_chol
    tol_active = solver.opts.active_set_tolerance_pn

    K = solver.K
    l = solver.l
    jac = solver.jac

    Π_curr = solver.Π[1]
    Π_next = solver.Π[2]
    π_curr = solver.π_v[1]
    π_next = solver.π_v[2]

    Qxx = solver.Qxx
    Qux = solver.Qux
    Quu = solver.Quu
    Qx  = solver.Qx
    Qu  = solver.Qu

    tmp_nn = solver.tmp_nn
    tmp_mn = solver.tmp_mn
    tmp_nm = solver.tmp_nm
    tmp_n1 = solver.tmp_n1
    tmp_n2 = solver.tmp_n2
    tmp_m  = solver.tmp_m

    # — Terminal stage: ΠN = QN + ρ_primal*I, πN = 0 —
    E = TO.Expansion{T}(n, m)
    stage_objective_expansion!(E, obj, Z, N, solver.exp_cache)
    copyto!(Π_next, E.Q)
    for i = 1:n
        Π_next[i,i] += ρ_primal
    end
    fill!(π_next, zero(T))

    # Add state constraint penalties at terminal stage
    μ_state = T(1e6)  # large penalty for state constraint projection
    _add_state_constraint_penalties!(Π_next, π_next, conSet, Z, N, tol_active, μ_state)

    # — Backward sweep: k = N-1 down to 1 —
    for k = (N-1):-1:1
        # 1. Dynamics Jacobian at stage k
        RD.discrete_jacobian!(QUAD, jac, model, Z[k], solver.fd_cache)
        Ak = view(jac, :, 1:n)       # n×n
        Bk = view(jac, :, n+1:n+m)   # n×m

        # 2. Dynamics residual
        x_next_pred = RD.discrete_dynamics(QUAD, model, Z[k])
        # dk = x_{k+1} - f(x_k, u_k)
        dk = state(Z[k+1]) - x_next_pred  # SVector

        # 3. Cost Hessian at stage k (original objective, no AL terms)
        stage_objective_expansion!(E, obj, Z, k, solver.exp_cache)
        Qk = E.Q   # n×n (SizedMatrix view into E.hess)
        Rk = E.R   # m×m
        Mk = E.H   # m×n (cross-term: ∂²/∂u∂x)

        # 4. Form Q-function (action-value expansion)
        # Q̃xx = Qk + Ak'*Π_next*Ak + ρ_primal*I
        mul!(tmp_nn, Transpose(Ak), Π_next)
        mul!(Qxx, tmp_nn, Ak)
        Qxx .+= Qk
        for i = 1:n
            Qxx[i,i] += ρ_primal
        end

        # Q̃ux = Mk + Bk'*Π_next*Ak
        mul!(tmp_mn, Transpose(Bk), Π_next)
        mul!(Qux, tmp_mn, Ak)
        Qux .+= Mk

        # Q̃uu = Rk + Bk'*Π_next*Bk + ρ_chol*I
        mul!(Quu, tmp_mn, Bk)
        Quu .+= Rk
        for i = 1:m
            Quu[i,i] += ρ_chol
        end

        # Q̃x = -Ak'*Π_next*dk + Ak'*π_next  (gradient from dynamics residual)
        mul!(Qx, Transpose(Ak), π_next)
        mul!(tmp_n1, Π_next, dk)
        mul!(Qx, Transpose(Ak), tmp_n1, -1.0, 1.0)

        # Q̃u = -Bk'*Π_next*dk + Bk'*π_next
        mul!(Qu, Transpose(Bk), π_next)
        mul!(Qu, Transpose(Bk), tmp_n1, -1.0, 1.0)

        # 5. Collect active non-state constraints at this stage
        # Only control/stage constraints go through Schur complement
        Cx_all, Cu_all, e_all = _collect_nonstate_constraints_for_stage(
            conSet, Z, k, tol_active, n, m)

        # 6. Compute gains with constraint handling
        _compute_constrained_gains!(K[k], l[k], Qxx, Qux, Quu, Qx, Qu,
            Cx_all, Cu_all, e_all, tmp_nn, tmp_mn, tmp_nm, tmp_m, n, m)

        # 7. Update cost-to-go: Πk, πk
        # Π_curr = Q̃xx + K'*Q̃uu*K + K'*Q̃ux + Q̃ux'*K
        mul!(tmp_mn, Quu, K[k])
        mul!(Π_curr, Transpose(K[k]), tmp_mn)
        Π_curr .+= Qxx
        mul!(Π_curr, Transpose(K[k]), Qux, 1.0, 1.0)
        mul!(Π_curr, Transpose(Qux), K[k], 1.0, 1.0)
        # Symmetrize
        for i = 1:n, j = i+1:n
            avg = 0.5*(Π_curr[i,j] + Π_curr[j,i])
            Π_curr[i,j] = avg
            Π_curr[j,i] = avg
        end

        # π_curr = Q̃x + K'*Q̃uu*l + K'*Q̃u + Q̃ux'*l
        mul!(tmp_m, Quu, l[k])
        mul!(π_curr, Transpose(K[k]), tmp_m)
        π_curr .+= Qx
        mul!(π_curr, Transpose(K[k]), Qu, 1.0, 1.0)
        mul!(π_curr, Transpose(Qux), l[k], 1.0, 1.0)

        # 8. Add state constraint penalties at stage k into cost-to-go
        _add_state_constraint_penalties!(Π_curr, π_curr, conSet, Z, k, tol_active, μ_state)

        # Swap Π, π for next iteration
        Π_curr, Π_next = Π_next, Π_curr
        π_curr, π_next = π_next, π_curr
    end

    # Store back (in case of odd number of swaps)
    solver.Π[1] .= Π_curr
    solver.Π[2] .= Π_next
    solver.π_v[1] .= π_curr
    solver.π_v[2] .= π_next

    return nothing
end

# ——————————————————————————————————————————————————————————
#  Constraint collection helpers
# ——————————————————————————————————————————————————————————

"""
Add penalty terms for active state constraints at stage k into the cost-to-go.

For each active state constraint Cx*x = -e (violation e = c(x)):
    Π += μ * Cx' * Cx
    π -= μ * Cx' * e

This avoids pushing state constraints back through dynamics (which causes
over-constrained stages when p > m) and instead softly penalizes them
in the Riccati cost-to-go.
"""
function _add_state_constraint_penalties!(
        Π::Matrix, π_v::Vector,
        conSet::MemEffConstraintSet, Z::Traj, k::Int, tol::Float64, μ::Real)

    for conval in conSet.convals
        !(conval.con isa TO.StateConstraint) && continue
        i = stage_index(conval.inds, k)
        i == 0 && continue

        sense = TO.sense(conval.con)
        c = conval.vals[i]

        if sense == TO.Equality()
            fill!(conval.jac, 0.0)
            TO.jacobian!(conval.jac, conval.con, Z[k])
            Cx = conval.jac  # p×n
            # Π += μ * Cx' * Cx
            mul!(Π, Transpose(Cx), Cx, μ, 1.0)
            # π += μ * Cx' * e  (gradient of penalty (1/2)μ||Cx*δx+e||²)
            mul!(π_v, Transpose(Cx), c, μ, 1.0)
        elseif sense == TO.Inequality()
            λ_k = conval.λ[i]
            active = (c .>= -tol) .| (λ_k .> 0)
            if any(active)
                fill!(conval.jac, 0.0)
                TO.jacobian!(conval.jac, conval.con, Z[k])
                Cx_active = conval.jac[active, :]
                c_active = c[active]
                mul!(Π, Transpose(Cx_active), Cx_active, μ, 1.0)
                mul!(π_v, Transpose(Cx_active), c_active, μ, 1.0)
            end
        end
    end
    return nothing
end

"""
Collect active non-state constraints (ControlConstraint, StageConstraint) at stage k.
These are the only constraints that go through the Schur complement in the Riccati
backward pass, ensuring p ≤ m (number of constraints ≤ number of controls).

Returns (Cx_all, Cu_all, e_all) stacked.
"""
function _collect_nonstate_constraints_for_stage(
        conSet::MemEffConstraintSet, Z::Traj, k::Int, tol::Float64,
        n::Int, m::Int)

    Cx_blocks = Matrix{Float64}[]
    Cu_blocks = Matrix{Float64}[]
    e_blocks = Vector{Float64}[]

    for conval in conSet.convals
        # Skip state-only constraints (handled via penalty in cost-to-go)
        conval.con isa TO.StateConstraint && continue

        i = stage_index(conval.inds, k)
        i == 0 && continue

        sense = TO.sense(conval.con)
        c = conval.vals[i]

        if sense == TO.Equality()
            fill!(conval.jac, 0.0)
            TO.jacobian!(conval.jac, conval.con, Z[k])
            cx, cu = _split_jacobian(conval.jac, conval.con, n, m)
            push!(Cx_blocks, copy(cx))
            push!(Cu_blocks, copy(cu))
            push!(e_blocks, copy(c))
        elseif sense == TO.Inequality()
            λ_k = conval.λ[i]
            active = (c .>= -tol) .| (λ_k .> 0)
            if any(active)
                fill!(conval.jac, 0.0)
                TO.jacobian!(conval.jac, conval.con, Z[k])
                cx, cu = _split_jacobian(conval.jac, conval.con, n, m)
                push!(Cx_blocks, cx[active, :])
                push!(Cu_blocks, cu[active, :])
                push!(e_blocks, c[active])
            end
        end
    end

    if isempty(Cx_blocks)
        return zeros(0, n), zeros(0, m), zeros(0)
    end
    return vcat(Cx_blocks...), vcat(Cu_blocks...), vcat(e_blocks...)
end

"""Split a constraint Jacobian into state and control parts."""
function _split_jacobian(jac::AbstractMatrix, con, n::Int, m::Int)
    if con isa TO.ControlConstraint
        return zeros(size(jac,1), n), Matrix(jac)
    elseif con isa TO.StageConstraint
        cx = Matrix(jac[:, 1:n])
        cu = Matrix(jac[:, n+1:n+m])
        return cx, cu
    else
        # Fallback: assume [x; u] layout
        cx = Matrix(jac[:, 1:min(n, size(jac,2))])
        cu = size(jac,2) > n ? Matrix(jac[:, n+1:n+m]) : zeros(size(jac,1), m)
        return cx, cu
    end
end

# ——————————————————————————————————————————————————————————
#  Constrained gain computation
# ——————————————————————————————————————————————————————————
"""
Compute feedback gains K, l with active constraints.

Without constraints:
    δu = -Q̃uu⁻¹*(Q̃ux*δx + Q̃u)

With constraints Cx*δx + Cu*δu = -e:
    Solve the mini-KKT:
    [Q̃uu  Cu'] [δu]   [-Q̃ux*δx - Q̃u]
    [Cu    0 ] [λ ] = [-Cx*δx - e   ]
"""
function _compute_constrained_gains!(
        K::Matrix, l::Vector,
        Qxx, Qux, Quu, Qx, Qu,
        Cx, Cu, e_con,
        tmp_nn, tmp_mn, tmp_nm, tmp_m,
        n::Int, m::Int)

    p = size(Cx, 1)  # number of active constraints

    if p == 0
        # Unconstrained: K = -Quu⁻¹*Qux, l = -Quu⁻¹*Qu
        Quu_fact = cholesky(Symmetric(Quu))
        K .= -(Quu_fact \ Qux)
        l .= -(Quu_fact \ Qu)
        return nothing
    end

    # Factorize Q̃uu
    Quu_fact = cholesky(Symmetric(Quu))

    # Unconstrained gains
    Ku = -(Quu_fact \ Qux)   # m×n
    lu = -(Quu_fact \ Qu)    # m

    # Effective constraint after substituting unconstrained step
    # C̃ = Cx + Cu*Ku  (effective state constraint)
    C_tilde = Cx + Cu * Ku  # p×n

    # ẽ = e + Cu*lu
    e_tilde = e_con + Cu * lu  # p

    # Schur complement: S = Cu*Quu⁻¹*Cu'  (p×p)
    Y = Quu_fact \ Matrix(Cu')  # m×p
    S = Cu * Y                    # p×p

    # Regularize S for numerical stability
    for i = 1:p
        S[i,i] += 1e-8
    end

    # Solve S*[Kλ; kλ] = [C̃; ẽ]
    S_fact = cholesky!(Symmetric(S))
    Kλ = S_fact \ C_tilde  # p×n
    kλ = S_fact \ e_tilde  # p

    # Constrained gains
    # K = Ku - Y*Kλ
    K .= Ku
    mul!(K, Y, Kλ, -1.0, 1.0)

    # l = lu - Y*kλ
    l .= lu
    mul!(l, Y, kλ, -1.0, 1.0)

    return nothing
end

# ——————————————————————————————————————————————————————————
#  Forward rollout
# ——————————————————————————————————————————————————————————
"""
Forward rollout using computed gains K[k], l[k].

    x̄₀ = x₀ (fixed initial state)
    for k = 1:N-1:
        δxₖ = x̄ₖ - xₖ
        δuₖ = K[k]*δxₖ + α*l[k]
        ūₖ = uₖ + δuₖ
        x̄ₖ₊₁ = f(x̄ₖ, ūₖ)

The dynamics are exactly satisfied by construction (rollout).
"""
function _memeff_forward_rollout!(solver::MemEffProjectedNewtonSolver{T,QUAD,n,m}, α::T) where {T,QUAD,n,m}
    Z = solver.Z
    Z̄ = solver.Z̄
    N = solver.Nsteps
    K = solver.K
    l = solver.l
    model = solver.model

    # Initial state
    Z̄[1].z = Z[1].z  # copy full knot point (preserves control at k=1)

    δx = solver.tmp_n1
    δu = solver.tmp_m

    for k = 1:N-1
        # δx = x̄_k - x_k
        for i = 1:n
            δx[i] = Z̄[k].z[i] - Z[k].z[i]
        end

        # δu = K[k]*δx + α*l[k]
        mul!(δu, K[k], δx)
        δu .+= α .* l[k]

        # Apply control
        u_new = control(Z[k]) + SVector{m}(δu)
        RD.set_control!(Z̄[k], u_new)

        # Rollout dynamics
        x_next = RD.discrete_dynamics(QUAD, model, Z̄[k])
        Z̄[k+1].z = [x_next; control(Z[k+1])]
    end

    return nothing
end
