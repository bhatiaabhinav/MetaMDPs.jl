using Random
using MDPs
import MDPs: state_space, action_space, action_meaning, state, action, reward, reset!, step!, in_absorbing_state, visualize, factory_reset!

include("metamdp_observation.jl")

mutable struct MetaMDP{S, A} <: AbstractMDP{Vector{Float32}, A}
    const tasks  # Some iterable. Need not be finite. Can be a generator.
    const task_horizon::Real
    const horizon::Real
    task::AbstractMDP{S, A}
    tasks_iterator_state
    const meta_observation::MetaMDPObservation
    const meta_observation_compiled::Vector{Float32}
    action::A
    reward::Float64
    task_episode_steps::Int
    steps::Int
    const 𝕊::VectorSpace{Float32}

    function MetaMDP(tasks, horizon::Real=Inf, cat_latest_experience=false, cat_time=false; add_time_to_indices::Union{Nothing, Tuple{Int, Int}}=nothing, task_horizon::Real=Inf)

        if cat_time && !isnothing(add_time_to_indices)
            @warn "cat_time and add_time_to_indices are mutually exclusive. Ignoring add_time_to_indices." maxlog=1
            add_time_to_indices = nothing
        end
        
        task_horizon = min(task_horizon, horizon)
        
        next = iterate(tasks)
        @assert !isnothing(next) 
        task, iter_state = next
        tasks_iterator_state = nothing
        
        sspace, aspace = state_space(task), action_space(task)
        sdim, adim = size(sspace, 1), size(aspace, 1)
        S, A = eltype(sspace), eltype(aspace)
        meta_obs = MetaMDPObservation(undef, sdim, adim, task_horizon, horizon, cat_latest_experience, cat_time, add_time_to_indices)
        reset_all!(meta_obs)
        meta_obs_compiled = zeros(Float32, length(meta_obs))
        a = action(task)
        r = reward(task)
        sspace_lows = S == Int ? zeros(Int, sdim) : sspace.lows
        sspace_highs = S == Int ? ones(Int, sdim) : sspace.highs
        aspace_lows = A == Int ? zeros(Int, adim) : aspace.lows
        aspace_highs = A == Int ? ones(Int, adim) : aspace.highs
        lows, highs = get_range(meta_obs, sspace_lows, sspace_highs, aspace_lows, aspace_highs)
        𝕊 = VectorSpace{Float32}(lows, highs)

        return new{S, A}(tasks, task_horizon, horizon, task, tasks_iterator_state, meta_obs, meta_obs_compiled, a, r, 0, 0, 𝕊)
    end
end

function factory_reset!(mm::MetaMDP)
    mm.task, mm.tasks_iterator_state = iterate(mm.tasks)
    mm.tasks_iterator_state = nothing
    reset_all!(mm.meta_observation)
    fill!(mm.meta_observation_compiled, 0)
    mm.action = action(mm.task)
    mm.reward = reward(mm.task)
    mm.task_episode_steps = 0
    mm.steps = 0
    nothing
end


state_space(mm::MetaMDP) = mm.𝕊
action_space(mm::MetaMDP) = action_space(mm.task)
action_meaning(mm::MetaMDP{S,A}, a::A) where {S,A} = action_meaning(mm.task, a)


state(mm::MetaMDP) = mm.meta_observation_compiled
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

    mm.task, mm.tasks_iterator_state = next  # switch to a new task
    mm.task_episode_steps = 0
    mm.steps = 0
    factory_reset!(mm.task)
    reset!(mm.task; rng=rng)
    @debug "Sampled new task" mm.task

    reset_all!(mm.meta_observation)
    set_state!(mm.meta_observation, state(mm.task))
    compile_to_vector!(mm.meta_observation, mm.meta_observation_compiled)

    mm.action = action(mm.task)
    mm.reward = reward(mm.task)

    return nothing
end

function step!(mm::MetaMDP{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    @assert a ∈ action_space(mm)

    if mm.meta_observation.cat_latest_experience
        set_state!(mm.meta_observation.latest_experience, state(mm.task))
    end
    
    step!(mm.task, a; rng=rng)
    mm.task_episode_steps += 1
    mm.steps += 1
    r = reward(mm.task)
    s′ = state(mm.task)
    term = in_absorbing_state(mm.task)
    trunc = mm.task_episode_steps >= mm.task_horizon || truncated(mm.task)

    if mm.meta_observation.cat_latest_experience
        set_action!(mm.meta_observation.latest_experience, a)
        set_reward!(mm.meta_observation.latest_experience, r)
        set_next_state!(mm.meta_observation.latest_experience, s′)
        set_terminated!(mm.meta_observation.latest_experience, term)
        set_truncated!(mm.meta_observation.latest_experience, trunc)
    end

    mm.action = a
    mm.reward = reward(mm.task)

    if term || trunc
        reset!(mm.task; rng=rng)
        mm.task_episode_steps = 0
    end

    set_state!(mm.meta_observation, state(mm.task))
    set_steps!(mm.meta_observation, mm.steps)
    set_task_episode_steps!(mm.meta_observation, mm.task_episode_steps)
    compile_to_vector!(mm.meta_observation, mm.meta_observation_compiled)

    nothing
end


in_absorbing_state(mm::MetaMDP)::Bool =  mm.steps >= mm.horizon  # it's not a continual task if given a fixed `horizon`

visualize(mm::MetaMDP, args...; kwargs...) = visualize(mm.task, args...; kwargs...)
