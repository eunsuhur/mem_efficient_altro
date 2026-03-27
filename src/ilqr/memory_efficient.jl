struct MemoryEfficientBackwardpassScratch{T}
    jac::Matrix{T}
end

const MEMORY_EFFICIENT_BACKWARDPASS_SCRATCH = Ref{Any}(nothing)

@inline function stage_index(inds::Vector{Int}, k::Int)
    i = searchsortedfirst(inds, k)
    return i <= length(inds) && inds[i] == k ? i : 0
end

@inline function reset_expansion!(E::TO.Expansion)
    fill!(E.grad, zero(eltype(E.grad)))
    fill!(E.hess, zero(eltype(E.hess)))
    return E
end

function stage_objective_expansion!(E::TO.Expansion, obj::TO.Objective, Z::RD.Traj, k::Int, cache)
    reset_expansion!(E)
    TO.gradient!(E, obj.cost[k], Z[k], cache)
    TO.hessian!(E, obj.cost[k], Z[k], cache)
    dt_x = k < length(Z) ? Z[k].dt : one(Z[k].dt)
    dt_u = k < length(Z) ? Z[k].dt : zero(Z[k].dt)
    E.q .*= dt_x
    E.r .*= dt_u
    E.Q .*= dt_x
    E.R .*= dt_u
    E.H .*= dt_u
    return E
end

stage_cost_expansion!(E::TO.Expansion, obj::TO.Objective, Z::RD.Traj, k::Int, cache) =
    stage_objective_expansion!(E, obj, Z, k, cache)

function get_backwardpass_scratch!(::Type{T}, n::Int, m::Int) where T
    scratch = MEMORY_EFFICIENT_BACKWARDPASS_SCRATCH[]
    if scratch isa MemoryEfficientBackwardpassScratch{T} && size(scratch.jac) == (n, n + m)
        return scratch.jac
    end
    scratch = MemoryEfficientBackwardpassScratch{T}(zeros(T, n, n + m))
    MEMORY_EFFICIENT_BACKWARDPASS_SCRATCH[] = scratch
    return scratch.jac
end

function backwardpass_memory_efficient!(solver::iLQRSolver{T,QUAD,L,O,n,n̄,m}) where {T,QUAD<:RD.QuadratureRule,L,O,n,n̄,m}
    model = solver.model
    Z = solver.Z
    K = solver.K
    d = solver.d
    S = solver.S
    Quu_reg = solver.Quu_reg
    Qux_reg = solver.Qux_reg

    jac = get_backwardpass_scratch!(T, n, m)
    fdx = view(jac, :, 1:n)
    fdu = view(jac, :, (n + 1):(n + m))

    E_stage = solver.E[1]
    E_terminal = solver.E[2]
    stage_cost_expansion!(E_terminal, solver.obj, Z, solver.N, solver.exp_cache)

    S_next = S[2]
    S_curr = S[1]
    S_next.Q .= E_terminal.Q
    S_next.q .= E_terminal.q

    ΔV = SVector(zero(T), zero(T))
    k = solver.N - 1
    while k > 0
        RD.discrete_jacobian!(QUAD, jac, model, Z[k], solver.cache)

        cost_exp = stage_cost_expansion!(E_stage, solver.obj, Z, k, solver.exp_cache)
        Q = solver.Q_tmp
        _calc_Q!(Q, cost_exp, S_next, fdx, fdu, S_curr)

        Quu_reg .= Q.R
        if solver.opts.bp_reg_type == :state
            _bp_reg!(Quu_reg, Qux_reg, Q, fdx, fdu, solver.ρ[1], :state)
        else
            Qux_reg .= Q.H
            @inbounds for j = 1:m
                Quu_reg[j, j] += solver.ρ[1]
            end
        end

        if solver.opts.bp_reg
            vals = eigvals(Hermitian(Quu_reg))
            if minimum(vals) <= 0
                @warn "Backward pass regularized"
                regularization_update!(solver, :increase)
                k = solver.N - 1
                ΔV = SVector(zero(T), zero(T))
                continue
            end
        end

        _calc_gains!(K[k], d[k], Quu_reg, Qux_reg, Q.r)
        ΔV += _calc_ctg!(S_curr, Q, K[k], d[k])
        S_next, S_curr = S_curr, S_next
        k -= 1
    end

    regularization_update!(solver, :decrease)
    return ΔV
end

function step_memory_efficient!(solver::iLQRSolver, J)
    to = solver.stats.to
    @timeit_debug to "backward pass" ΔV = backwardpass_memory_efficient!(solver)
    @timeit_debug to "forward pass" forwardpass!(solver, ΔV, J)
end
