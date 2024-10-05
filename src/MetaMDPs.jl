module MetaMDPs

using Random
using MDPs
import MDPs: state_space, action_space, action_meaning, state, action, reward, reset!, step!, in_absorbing_state, visualize, factory_reset!

export MetaMDPincludePrevState, MetaMDP


mutable struct MetaMDP{S, A} <: AbstractMDP{Vector{Float32}, A}
    tasks  # Some iterable. Need not be finite. Can be a generator.
    task_horizon::Real
    horizon::Real
    include_time_context::Symbol # can be :none", :concat, :add
    add_time_to_indices::Union{Nothing, Tuple{Int, Int}}
    task::AbstractMDP{S, A}
    tasks_iterator_state
    state::Vector{Float32}
    action::A
    reward::Float64
    task_episode_steps::Int
    steps::Int
    𝕊::VectorSpace{Float32}

    function MetaMDP(tasks, horizon::Real=Inf, include_time_context=:none; add_time_to_indices::Union{Nothing, Tuple{Int, Int}}=(1, 2), task_horizon::Real=Inf)
        @assert include_time_context ∈ (:none, :concat, :add)  "Invalid time context method $include_time_context. Must be one of `:none`, `:concat`, `:add`"
        next = iterate(tasks)
        @assert !isnothing(next) 
        task, iter_state = next
        sspace, aspace = state_space(task), action_space(task)
        S, A = eltype(sspace), eltype(aspace)
        m = size(sspace, 1)
        _m = include_time_context == :concat ? m + 2 : m
        if S == Int
            𝕊 = VectorSpace{Float32}(0, 1, (_m, ))
        else
            lows = Float32.(sspace.lows) |> copy
            highs = Float32.(sspace.highs) |> copy
            if include_time_context == :concat
                lows = vcat(zeros(Float32, 2), lows)
                highs = vcat(ones(Float32, 2), highs)
            elseif include_time_context == :add
                highs[add_time_to_indices[1]] += 1
                highs[add_time_to_indices[2]] += 1
            end
            𝕊 = VectorSpace{Float32}(lows, highs)
        end
        task_horizon = min(task_horizon, horizon)
        # println("Task horizon: ", task_horizon, " Horizon: ", horizon)
        return new{S, A}(tasks, task_horizon, horizon, include_time_context, add_time_to_indices, task, nothing, zeros(Float32, _m), action(task), reward(task), 0, 0, 𝕊)
    end
end

function factory_reset!(mm::MetaMDP)
    mm.task, mm.tasks_iterator_state = iterate(mm.tasks)
    mm.tasks_iterator_state = nothing
    mm.steps = 0
    mm.task_episode_steps = 0
    nothing
end


state_space(mm::MetaMDP) = mm.𝕊
action_space(mm::MetaMDP) = action_space(mm.task)
action_meaning(mm::MetaMDP{S,A}, a::A) where {S,A} = action_meaning(mm.task, a)


state(mm::MetaMDP) = mm.state
action(mm::MetaMDP) = mm.action
reward(mm::MetaMDP) = mm.reward

function update_state!(mm::MetaMDP{Int, A})::Nothing where {A}
    mm.state .= 0
    if mm.include_time_context == :concat
        mm.state[1] = mm.steps / mm.horizon
        mm.state[2] = mm.task_episode_steps / mm.task_horizon
        mm.state[2+state(mm.task)] = 1
    elseif mm.include_time_context == :add
        mm.state[state(mm.task)] = 1
        mm.state[mm.add_time_to_indices[1]] += mm.steps / mm.horizon
        mm.state[mm.add_time_to_indices[2]] += mm.task_episode_steps / mm.task_horizon
    elseif mm.include_time_context == :none
        mm.state[state(mm.task)] = 1
    else
        error("Invalid time context method")
    end
    nothing
end

function update_state!(mm::MetaMDP{Vector{Float32}, A})::Nothing where {A}
    # println(mm.steps, " ", mm.horizon, " ", mm.steps / mm.horizon)
    if mm.include_time_context == :concat
        mm.state[1] = mm.steps / mm.horizon
        mm.state[2] = mm.task_episode_steps / mm.task_horizon
        mm.state[3:end] .= state(mm.task)
    elseif mm.include_time_context == :add
        mm.state .= state(mm.task)
        mm.state[mm.add_time_to_indices[1]] += mm.steps / mm.horizon
        mm.state[mm.add_time_to_indices[2]] += mm.task_episode_steps / mm.task_horizon
    elseif mm.include_time_context == :none
        mm.state .= state(mm.task)
    else
        error("Invalid time context method")
    end
    nothing
end

function reset!(mm::MetaMDP{S, A}; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
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

function step!(mm::MetaMDP{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
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


in_absorbing_state(mm::MetaMDP)::Bool =  mm.steps >= mm.horizon  # it's not a continual task if given a fixed `horizon`

visualize(mm::MetaMDP, args...; kwargs...) = visualize(mm.task, args...; kwargs...)



mutable struct MetaMDPincludePrevState{S, A} <: AbstractMDP{Vector{Float32}, A}
    tasks  # Some iterable. Need not be finite. Can be a generator.
    task_horizon::Real
    horizon::Real
    include_time_context::Symbol # can be :none", :concat, :add
    include_prev_s′::Bool
    inner_sspace_length::Int
    add_time_to_indices::Union{Nothing, Tuple{Int, Int}}
    task::AbstractMDP{S, A}
    tasks_iterator_state
    state::Vector{Float32}
    action::A
    reward::Float64
    task_episode_steps::Int
    steps::Int
    𝕊::VectorSpace{Float32}

    function MetaMDPincludePrevState(tasks, horizon::Real=Inf, include_time_context=:none; add_time_to_indices::Union{Nothing, Tuple{Int, Int}}=(1, 2), task_horizon::Real=Inf, include_prev_s′=false)
        @assert include_time_context ∈ (:none, :concat, :add)  "Invalid time context method $include_time_context. Must be one of `:none`, `:concat`, `:add`"
        next = iterate(tasks)
        @assert !isnothing(next) 
        task, iter_state = next
        sspace, aspace = state_space(task), action_space(task)
        S, A = eltype(sspace), eltype(aspace)
        m = size(sspace, 1)
        if include_prev_s′
            m += m + 1  # prev_s′ and a flag to mark if it is an absorbing state
        end
        _m = include_time_context == :concat ? m + 2 : m
        if S == Int
            𝕊 = VectorSpace{Float32}(0, 1, (_m, ))
        else
            lows = Float32.(sspace.lows) |> copy
            highs = Float32.(sspace.highs) |> copy
            if include_prev_s′
                lows = vcat(lows, lows, zeros(Float32, 1))
                highs = vcat(highs, highs, ones(Float32, 1))
            end
            if include_time_context == :concat
                lows = vcat(zeros(Float32, 2), lows)
                highs = vcat(ones(Float32, 2), highs)
            elseif include_time_context == :add
                highs[add_time_to_indices[1]] += 1
                highs[add_time_to_indices[2]] += 1
            end
            𝕊 = VectorSpace{Float32}(lows, highs)
        end
        task_horizon = min(task_horizon, horizon)
        return new{S, A}(tasks, task_horizon, horizon, include_time_context, include_prev_s′,  size(sspace, 1), add_time_to_indices, task, nothing, zeros(Float32, _m), action(task), reward(task), 0, 0, 𝕊)
    end
end

function factory_reset!(mm::MetaMDPincludePrevState)
    mm.task, mm.tasks_iterator_state = iterate(mm.tasks)
    mm.tasks_iterator_state = nothing
    mm.steps = 0
    mm.task_episode_steps = 0
    nothing
end


state_space(mm::MetaMDPincludePrevState) = mm.𝕊
action_space(mm::MetaMDPincludePrevState) = action_space(mm.task)
action_meaning(mm::MetaMDPincludePrevState{S,A}, a::A) where {S,A} = action_meaning(mm.task, a)


state(mm::MetaMDPincludePrevState) = mm.state
action(mm::MetaMDPincludePrevState) = mm.action
reward(mm::MetaMDPincludePrevState) = mm.reward

function update_state!(mm::MetaMDPincludePrevState{Int, A}; prev_s′::Union{Nothing, Int}=nothing, is_prev_s′_absorbing::Bool=false)::Nothing where {A}
    mm.state .= 0
    if mm.include_time_context == :concat
        mm.state[1] = mm.steps / mm.horizon
        mm.state[2] = mm.task_episode_steps / mm.task_horizon
        mm.state[2+state(mm.task)] = 1
        if mm.include_prev_s′
            mm.state[2+mm.inner_sspace_length+prev_s′] = 1
            mm.state[end] = Float32(is_prev_s′_absorbing)
        end
    elseif mm.include_time_context == :add
        mm.state[state(mm.task)] = 1
        mm.state[mm.add_time_to_indices[1]] += mm.steps / mm.horizon
        mm.state[mm.add_time_to_indices[2]] += mm.task_episode_steps / mm.task_horizon
        if mm.include_prev_s′
            mm.state[mm.inner_sspace_length+prev_s′] = 1
            mm.state[end] = Float32(is_prev_s′_absorbing)
        end
    elseif mm.include_time_context == :none
        mm.state[state(mm.task)] = 1
        if mm.include_prev_s′
            mm.state[mm.inner_sspace_length+prev_s′] = 1
            mm.state[end] = Float32(is_prev_s′_absorbing)
        end
    else
        error("Invalid time context method")
    end
    nothing
end

function update_state!(mm::MetaMDPincludePrevState{Vector{Float32}, A}; prev_s′::Union{Nothing, Vector{Float32}}=nothing, is_prev_s′_absorbing::Bool=false)::Nothing where {A}
    # println(mm.steps, " ", mm.horizon, " ", mm.steps / mm.horizon)
    if mm.include_time_context == :concat
        mm.state[1] = mm.steps / mm.horizon
        mm.state[2] = mm.task_episode_steps / mm.task_horizon
        mm.state[3:2+mm.inner_sspace_length] .= state(mm.task)
        if mm.include_prev_s′
            mm.state[2+mm.inner_sspace_length+1:end-1] .= prev_s′
            mm.state[end] = Float32(is_prev_s′_absorbing)
        end
    elseif mm.include_time_context == :add
        mm.state .= state(mm.task)
        mm.state[mm.add_time_to_indices[1]] += mm.steps / mm.horizon
        mm.state[mm.add_time_to_indices[2]] += mm.task_episode_steps / mm.task_horizon
        if mm.include_prev_s′
            mm.state[mm.inner_sspace_length+1:end-1] .= prev_s′
            mm.state[end] = Float32(is_prev_s′_absorbing)
        end
    elseif mm.include_time_context == :none
        mm.state .= state(mm.task)
        if mm.include_prev_s′
            mm.state[mm.inner_sspace_length+1:end-1] .= prev_s′
            mm.state[end] = Float32(is_prev_s′_absorbing)
        end
    else
        error("Invalid time context method")
    end
    nothing
end

function reset!(mm::MetaMDPincludePrevState{S, A}; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
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
    update_state!(mm; prev_s′= S==Int ? 1 : zeros(Float32, mm.inner_sspace_length), is_prev_s′_absorbing=true)
    mm.action = action(mm.task)
    mm.reward = reward(mm.task)
    return nothing
end

function step!(mm::MetaMDPincludePrevState{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    @assert a ∈ action_space(mm)
    step!(mm.task, a; rng=rng)
    mm.task_episode_steps += 1
    mm.steps += 1
    update_state!(mm; prev_s′=state(mm.task))
    mm.action = a
    mm.reward = reward(mm.task)
    if in_absorbing_state(mm.task) || mm.task_episode_steps >= mm.task_horizon
        prev_s′ = copy(state(mm.task))
        is_prev_s′_absorbing = in_absorbing_state(mm.task)
        reset!(mm.task; rng=rng)
        mm.task_episode_steps = 0
        update_state!(mm; prev_s′=prev_s′, is_prev_s′_absorbing=is_prev_s′_absorbing)
    end
    nothing
end


in_absorbing_state(mm::MetaMDPincludePrevState)::Bool =  mm.steps >= mm.horizon  # it's not a continual task if given a fixed `horizon`

visualize(mm::MetaMDPincludePrevState, args...; kwargs...) = visualize(mm.task, args...; kwargs...)

end # module MetaMDPs