module MetaMDPs

using Random
using MDPs
import MDPs: state_space, action_space, action_meaning, state, action, reward, reset!, step!, in_absorbing_state, visualize, factory_reset!

export MetaMDP, MetaMDPwithTimeContext

mutable struct MetaMDP{S, A} <: AbstractMDP{S, A}
    tasks  # Some iterable. Need not be finite. Can be a generator.
    task_horizon::Real
    task::AbstractMDP{S, A}
    tasks_iterator_state
    state::S
    action::A
    reward::Float64
    task_episode_steps::Int

    function MetaMDP(tasks; task_horizon::Real=Inf)
        next = iterate(tasks)
        @assert !isnothing(next) 
        task, iter_state = next
        sspace, aspace = state_space(task), action_space(task)
        S, A = eltype(sspace), eltype(aspace)
        return new{S, A}(tasks, task_horizon, task, nothing, state(task), action(task), reward(task), 0)
    end
end

function factory_reset!(mm::MetaMDP)
    mm.task, mm.tasks_iterator_state = iterate(mm.tasks)
    mm.tasks_iterator_state = nothing
    nothing
end


state_space(mm::MetaMDP) = state_space(mm.task)
action_space(mm::MetaMDP) = action_space(mm.task)
action_meaning(mm::MetaMDP{S,A}, a::A) where {S,A} = action_meaning(mm.task, a)


state(mm::MetaMDP) = mm.state
action(mm::MetaMDP) = mm.action
reward(mm::MetaMDP) = mm.reward

function reset!(mm::MetaMDP{S, A}; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    # println(mm.tasks_iterator_state)
    next = isnothing(mm.tasks_iterator_state) ? iterate(mm.tasks) : iterate(mm.tasks, mm.tasks_iterator_state)
    if isnothing(next)
        @warn "Iterated through all tasks. Resetting iterator. This may cause repetition of tasks, probably in the same sequence as before. This warning will be shown only once." maxlog=1
        next = iterate(mm.tasks)
    end
    factory_reset!(mm.task)  # factory_reset the outgoing task to free up memory
    mm.task, mm.tasks_iterator_state = next
    factory_reset!(mm.task)
    reset!(mm.task; rng=rng)
    @debug "Sampled new task" mm.task
    if S == Int
        mm.state = state(mm.task)
    else
        mm.state .= state(mm.task)
    end
    mm.action = action(mm.task)
    mm.reward = reward(mm.task)
    mm.task_episode_steps = 0
    return nothing
end

function step!(mm::MetaMDP{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    @assert a ∈ action_space(mm)
    step!(mm.task, a; rng=rng)
    mm.task_episode_steps += 1
    if S == Int
        mm.state = state(mm.task)
    else
        mm.state .= state(mm.task)
    end
    mm.action = a
    mm.reward = reward(mm.task)
    if in_absorbing_state(mm.task) || mm.task_episode_steps >= mm.task_horizon
        reset!(mm.task; rng=rng)
        mm.task_episode_steps = 0
        if S == Int
            mm.state = state(mm.task)
        else
            mm.state .= state(mm.task)
        end
    end
    nothing
end


in_absorbing_state(mm::MetaMDP)::Bool =  false  # it's a continual task

visualize(mm::MetaMDP, args...; kwargs...) = visualize(mm.task, args...; kwargs...)








mutable struct MetaMDPwithTimeContext{S, A} <: AbstractMDP{Vector{Float32}, A}
    tasks  # Some iterable. Need not be finite. Can be a generator.
    task_horizon::Real
    horizon::Real
    task::AbstractMDP{S, A}
    tasks_iterator_state
    state::Vector{Float32}
    action::A
    reward::Float64
    task_episode_steps::Int
    steps::Int
    𝕊::VectorSpace{Float32}

    function MetaMDPwithTimeContext(tasks, horizon::Int; task_horizon::Real=Inf)
        next = iterate(tasks)
        @assert !isnothing(next) 
        task, iter_state = next
        sspace, aspace = state_space(task), action_space(task)
        S, A = eltype(sspace), eltype(aspace)
        m = size(sspace, 1)
        if S == Int
            𝕊 = VectorSpace{Float32}(0, 1, (m+2,))
        else
            𝕊 = VectorSpace{Float32}(vcat(Float32.(sspace.lows), zeros(Float32, 2)), vcat(Float32.(sspace.highs), ones(Float32, 2)))
        end
        task_horizon = min(task_horizon, horizon)
        return new{S, A}(tasks, task_horizon, horizon, task, nothing, zeros(Float32, m+2), action(task), reward(task), 0, 0, 𝕊)
    end
end

function factory_reset!(mm::MetaMDPwithTimeContext)
    mm.task, mm.tasks_iterator_state = iterate(mm.tasks)
    mm.tasks_iterator_state = nothing
    nothing
end


state_space(mm::MetaMDPwithTimeContext) = mm.𝕊
action_space(mm::MetaMDPwithTimeContext) = action_space(mm.task)
action_meaning(mm::MetaMDPwithTimeContext{S,A}, a::A) where {S,A} = action_meaning(mm.task, a)


state(mm::MetaMDPwithTimeContext) = mm.state
action(mm::MetaMDPwithTimeContext) = mm.action
reward(mm::MetaMDPwithTimeContext) = mm.reward

function update_state!(mm::MetaMDPwithTimeContext{Int, A})::Nothing where {A}
    mm.state .= 0
    mm.state[state(mm.task)] = 1
    mm.state[end-1] = mm.task_episode_steps / mm.task_horizon
    mm.state[end] = mm.steps / mm.horizon
    nothing
end

function update_state!(mm::MetaMDPwithTimeContext{Vector{Float32}, A})::Nothing where {A}
    mm.state[1:end-2] .= state(mm.task)
    mm.state[end-1] = mm.task_episode_steps / mm.task_horizon
    mm.state[end] = mm.steps / mm.horizon
    nothing
end

function reset!(mm::MetaMDPwithTimeContext{S, A}; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    # println(mm.tasks_iterator_state)
    next = isnothing(mm.tasks_iterator_state) ? iterate(mm.tasks) : iterate(mm.tasks, mm.tasks_iterator_state)
    if isnothing(next)
        @warn "Iterated through all tasks. Resetting iterator. This may cause repetition of tasks, probably in the same sequence as before. This warning will be shown only once." maxlog=1
        next = iterate(mm.tasks)
    end
    factory_reset!(mm.task)  # factory_reset the outgoing task to free up memory
    mm.task, mm.tasks_iterator_state = next
    mm.task_episode_steps = 0
    mm.steps = 0
    factory_reset!(mm.task)
    reset!(mm.task; rng=rng)
    @debug "Sampled new task" mm.task
    update_state!(mm)
    mm.action = action(mm.task)
    mm.reward = reward(mm.task)
    return nothing
end

function step!(mm::MetaMDPwithTimeContext{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    @assert a ∈ action_space(mm)
    step!(mm.task, a; rng=rng)
    mm.task_episode_steps += 1
    mm.steps += 1
    update_state!(mm)
    mm.action = a
    mm.reward = reward(mm.task)
    if in_absorbing_state(mm.task) || mm.task_episode_steps >= mm.task_horizon
        reset!(mm.task; rng=rng)
        mm.task_episode_steps = 0
        update_state!(mm)
    end
    nothing
end


in_absorbing_state(mm::MetaMDPwithTimeContext)::Bool =  mm.steps >= mm.horizon  # it's not a continual task if given a fixed `horizon`

visualize(mm::MetaMDPwithTimeContext, args...; kwargs...) = visualize(mm.task, args...; kwargs...)

end # module MetaMDPs