

using JuMP
using PowerModels
using Ipopt # Assuming Ipopt is your solver
using MathOptInterface # MOI
using InfrastructureModels # Needed for _guard_objective_value, _guard_objective_bound
using JSON
const _PM = PowerModels # Alias for convenience, as used in your example
import PandaModels as _PdM # Assuming this alias is already here


function run_powermodels_opf(json_path)
    pm = _PdM.load_pm_from_json(json_path)
    active_powermodels_silence!(pm)
    pm = remove_extract_params!(pm)
    
    # NEW LOGIC HERE: Check for presence of target data
    # We'll use a specific key in pm.data to indicate if the projection OPF should be run
    # For example, check if any generator has "target_pg" or any bus has "target_vm"
    run_projection = false
    for (i, gen) in pm["gen"]
        if haskey(gen, "target_pg")
            run_projection = true
            break
        end
    end
    if !run_projection
        for (i, bus) in pm["bus"]
            if haskey(bus, "target_vm")
                run_projection = true
                break
            end
        end
    end

    solver = get_solver(pm)
    result = nothing # Initialize result

    if run_projection # If target data is present, run your custom projection OPF
        println("--- Running Custom Projection OPF ---")
        model = get_model(pm["pm_model"]) # e.g., ACPPowerModel
        result = _PM.solve_model(
            pm,
            model,
            solver,
            build_method = _build_projection_opf, # Call YOUR custom build method
            setting = Dict("output" => Dict("branch_flows" => true)),
        )
    else # Otherwise, run the standard OPF as before
        println("--- Running Standard OPF ---")
        cl = check_current_limit!(pm)
        if cl == 0
            pm = check_powermodels_data!(pm)
            result = _PM.solve_opf(
                pm,
                get_model(pm["pm_model"]), # Get model again
                solver,
                setting = Dict("output" => Dict("branch_flows" => true)),
            )
        else
            result = _PM._solve_opf_cl(
                pm,
                get_model(pm["pm_model"]), # Get model again
                solver,
                setting = Dict("output" => Dict("branch_flows" => true)),
            )
        end
    end

    return result
end




function _build_projection_opf(pm::_PM.AbstractPowerModel)
    println("\n--- Starting _build_projection_opf ---")
    println("Input pm.data keys: ", sort(collect(keys(pm.data))))
    println("Number of active buses: ", length(_PM.ref(pm, :bus)))
    println("Number of active generators: ", length(_PM.ref(pm, :gen)))
    println("Model type used in Julia: ", typeof(pm.model)) # Verify this is an AC model type

    # Standard PowerModels variable declarations
    _PM.variable_bus_voltage(pm) # Will create :vm and :va for ACPPowerModel
    _PM.variable_gen_power(pm)   # Will create :pg and :qg
    _PM.variable_branch_power(pm)
    _PM.variable_dcline_power(pm, bounded = false)

    println("--- After PowerModels variable declarations ---")
    println("pm.var keys (should include :vm and :pg): ", sort(collect(keys(pm.var))))

    # Add extra variables for projection
    vars_collection = JuMP.VariableRef[]
    x_hat_values = Float64[]

    # IMPORTANT: Ensure "target_pg" and "target_vm" are added to pm data in Python's pp_to_pm_callback
    for (i, gen) in _PM.ref(pm, :gen)
        if haskey(gen, "target_pg")
            if haskey(_PM.var(pm), :pg) && i in keys(_PM.var(pm, :pg))
                push!(vars_collection, _PM.var(pm, :pg, i))
                push!(x_hat_values, gen["target_pg"])
            else
                @warn "Generator $i has 'target_pg' but :pg variable not found for this index in PowerModels model. Skipping."
            end
        end
    end

    for (i, bus) in _PM.ref(pm, :bus)
        if haskey(bus, "target_vm")
            if haskey(_PM.var(pm), :vm) && i in keys(_PM.var(pm, :vm))
                push!(vars_collection, _PM.var(pm, :vm, i))
                push!(x_hat_values, bus["target_vm"])
            else
                @warn "Bus $i has 'target_vm' but :vm variable not found for this index in PowerModels model. Skipping."
            end
        end
    end

    N = length(vars_collection)

    # Always define r and aux
    _PM.variable(pm, :r, 0.0) # r >= 0
    if N > 0
        _PM.variable(pm, :aux, 1:N) # Define aux only if there are variables to link
    else
        _PM.variable(pm, :aux, 1:0) # Define an empty container
    end

    if N > 0
        for k in 1:N
            JuMP.@constraint(pm.model, _PM.var(pm, :aux, k) == vars_collection[k])
        end
        JuMP.@constraint(pm.model, sum((_PM.var(pm, :aux, k) - x_hat_values[k])^2 for k in 1:N) <= _PM.var(pm, :r)^2)
    else
        @warn "No 'target_pg' or 'target_vm' found for active components. Projection constraint will not be added."
        # Optionally, force r to zero if no targets to project onto
        # JuMP.@constraint(pm.model, _PM.var(pm, :r) == 0.0)
    end

    pm.ext[:projection_vars] = vars_collection
    pm.ext[:projection_targets] = x_hat_values
    pm.ext[:projection_N] = N

    # Call the custom objective function
    objective_projection(pm)

    # Standard PowerModels constraint declarations
    _PM.constraint_model_voltage(pm) # This is where vm_min/max constraints would be applied

    for i in _PM.ids(pm, :ref_buses)
        _PM.constraint_theta_ref(pm, i)
    end

    for i in _PM.ids(pm, :bus)
        _PM.constraint_power_balance(pm, i)
    end

    for (i, branch) in _PM.ref(pm, :branch)
        _PM.constraint_ohms_yt_from(pm, i)
        _PM.constraint_ohms_yt_to(pm, i)

        _PM.constraint_thermal_limit_from(pm, i)
        _PM.constraint_thermal_limit_to(pm, i)
    end

    for i in _PM.ids(pm, :dcline)
        _PM.constraint_dcline_power_losses(pm, i)
    end

    println("Model is built!")
end

# --- Your objective_projection function ---
function objective_projection(pm::_PM.AbstractPowerModel)
    println("\n--- Starting objective_projection ---")
    N = pm.ext[:projection_N]
    
    if N == 0
        @warn "No projection variables found. Setting objective to Min 0.0."
        return JuMP.@objective(pm.model, Min, 0.0)
    end

    r = _PM.var(pm, :r)
    println("objective done! Minimizing r = ", r)
    return JuMP.@objective(pm.model, Min, r)
end












# ---------- functions below are from pandamodels.jl/src/input/tools.jl ---------------

function get_solver(pm)

    optimizer = pm["pm_solver"]
    nl = pm["pm_nl_solver"]
    mip = pm["pm_mip_solver"]
    log_level = pm["pm_log_level"]
    time_limit = pm["pm_time_limit"]
    nl_time_limit = pm["pm_nl_time_limit"]
    mip_time_limit = pm["pm_mip_time_limit"]
    tol = pm["pm_tol"]

    if optimizer == "gurobi"
        solver = JuMP.optimizer_with_attributes(
            Gurobi.Optimizer,
            "TimeLimit" => time_limit,
            "OutputFlag" => log_level,
            "FeasibilityTol" => tol,
            "OptimalityTol" => tol,
        )
    end

    if optimizer == "ipopt"
        solver = JuMP.optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => log_level,
            "max_cpu_time" => time_limit,
            "tol" => tol,
        )
    end

    if optimizer == "juniper" && nl == "ipopt" && mip == "cbc"
        mip_solver = JuMP.optimizer_with_attributes(
            Cbc.Optimizer,
            "logLevel" => log_level,
            "seconds" => mip_time_limit,
        )
        nl_solver = JuMP.optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => log_level,
            "max_cpu_time" => nl_time_limit,
            "tol" => 1e-4,
        )
        solver = JuMP.optimizer_with_attributes(
            Juniper.Optimizer,
            "nl_solver" => nl_solver,
            "mip_solver" => mip_solver,
            "log_levels" => [],
            "time_limit" => time_limit,
        )
    end

    if optimizer == "juniper" && nl == "gurobi" && mip == "cbc"
        mip_solver = JuMP.optimizer_with_attributes(
            Cbc.Optimizer,
            "logLevel" => log_level,
            "seconds" => mip_time_limit,
        )
        nl_solver = JuMP.optimizer_with_attributes(
            Gurobi.Optimizer,
            "TimeLimit" => nl_time_limit,
            "FeasibilityTol" => tol,
            "OptimalityTol" => tol,
        )
        solver = JuMP.optimizer_with_attributes(
            Juniper.Optimizer,
            "nl_solver" => nl_solver,
            "mip_solver" => mip_solver,
            "log_levels" => [],
            "time_limit" => time_limit,
        )
    end

    if optimizer == "juniper" && nl == "gurobi" && mip == "gurobi"
        mip_solver = JuMP.optimizer_with_attributes(
            Gurobi.Optimizer,
            "TimeLimit" => mip_time_limit,
            "FeasibilityTol" => tol,
            "OptimalityTol" => tol,
        )
        nl_solver = JuMP.optimizer_with_attributes(
            Gurobi.Optimizer,
            "TimeLimit" => nl_time_limit,
            "FeasibilityTol" => tol,
            "OptimalityTol" => tol,
        )
        solver = JuMP.optimizer_with_attributes(
            Juniper.Optimizer,
            "nl_solver" => nl_solver,
            "mip_solver" => mip_solver,
            "log_levels" => [],
            "time_limit" => time_limit,
        )
    end

    if optimizer == "knitro"
        solver = JuMP.optimizer_with_attributes(KNITRO.Optimizer, "tol" => tol)
    end

    if optimizer == "cbc"
        solver = JuMP.optimizer_with_attributes(
            Cbc.Optimizer,
            "seconds" => time_limit,
            "tol" => tol,
        )
    end

    if optimizer == "scip"
        solver = JuMP.optimizer_with_attributes(SCIP.Optimizer, "tol" => tol)
    end

    return solver

end

function load_pm_from_json(json_path)
    pm = Dict()
    open(json_path, "r") do f
        pm = JSON.parse(f)
    end
    for (idx, gen) in pm["gen"]
        if gen["model"] == 1
            pm["gen"][idx]["cost"] = convert(Array{Float64,1}, gen["cost"])
        end
    end
    return pm
end


function get_model(model_type)
    s = Symbol(model_type)
    return getfield(_PM, s)
end

function extract_params!(pm)
    if haskey(pm, "user_defined_params")
        params = Dict{Symbol,Dict{String,Any}}()
        for key in keys(pm["user_defined_params"])
            params[Symbol(key)] = pm["user_defined_params"][key]
        end
        delete!(pm, "user_defined_params")
    end
    return params
end

function remove_extract_params!(pm)
    if haskey(pm, "user_defined_params")
        params = Dict{Symbol,Dict{String,Any}}()
        for key in keys(pm["user_defined_params"])
            params[Symbol(key)] = pm["user_defined_params"][key]
        end
        delete!(pm, "user_defined_params")
    end
    return pm
end

function check_powermodels_data!(pm)
    if pm["correct_pm_network_data"]
        _PM.correct_network_data!(pm)
    end
    if haskey(pm, "simplify_net")
        if pm["simplify_net"]
            _PM.simplify_network!(pm)
            _PM.deactivate_isolated_components!(pm)
            _PM.propagate_topology_status!(pm)
        end
    end
    return pm
end

function active_powermodels_silence!(pm)
    if pm["silence"]
        _PM.silence()
    end
end

function check_current_limit!(pm)
    cl = 0
    for (i, branch) in pm["branch"]
        if "c_rating_a" in keys(branch)
            cl += 1
        end
    end
    return cl
end

function set_pq_values_from_timeseries(pm)
    # This function iterates over multinetwork entries and sets p, q values
    # of loads and "sgens" (which are loads with negative P and Q values)
    steps = pm["time_series"]["to_time_step"]-pm["time_series"]["from_time_step"]
    mn = _PM.replicate(pm, steps)

    for (step, network) in mn["nw"]
        step_1=string(parse(Int64,step) - 1)
        load_ts = pm["time_series"]["load"]
        network = delete!(network, "user_defined_params")
        for (idx, load) in network["load"]
            if haskey(load_ts, idx)
                load["pd"] = load_ts[idx]["p_mw"][step_1]
                if haskey(load_ts[idx], "q_mvar")
                    load["qd"] = load_ts[idx]["q_mvar"][step_1]
                end
            end
        end

        gen_ts = pm["time_series"]["gen"]
        for (idx, gen) in network["gen"]
            if haskey(gen_ts, idx)
                gen["pg"] = gen_ts[idx]["p_mw"][step_1]
                if haskey(gen_ts[idx], "max_p_mw")
                    gen["pmax"] = gen_ts[idx]["max_p_mw"][step_1]
                else
                    gen["pmax"] = gen_ts[idx]["p_mw"][step_1]
                end
                if haskey(gen_ts[idx], "min_p_mw")
                    gen["pmin"] = gen_ts[idx]["min_p_mw"][step_1]
                else
                    gen["pmin"] = gen_ts[idx]["p_mw"][step_1]
                end

                if haskey(gen_ts[idx], "max_q_mvar")
                    gen["qmax"] = gen_ts[idx]["max_q_mvar"][step_1]
                end
                if haskey(gen_ts[idx], "min_q_mvar")
                    gen["qmin"] = gen_ts[idx]["min_q_mvar"][step_1]
                end
                if haskey(gen_ts[idx], "q_mvar")
                    gen["qg"] = gen_ts[idx]["q_mvar"][step_1]
                end
            end
        end
    end
    return mn
end

# ---------------------- until here -------------------



# Make sure to include PowerModels if not already imported by PandaModels' wrapper
import PowerModels
const _PM = PowerModels # Alias for convenience

# You might need to import JuMP if not implicitly available or within a PowerModels context
import JuMP