using JuMP
using PowerModels
using Ipopt # Assuming Ipopt is your solver
using MathOptInterface # MOI
using InfrastructureModels # Needed for _guard_objective_value, _guard_objective_bound
using JSON
const _PM = PowerModels # Alias for convenience, as used in your example
import PandaModels as _PdM # Assuming this alias is already here


# You will also need a wrapper function for this new build method, similar to solve_projection_opf
function solve_voltage_projection_opf(file, model_type::Type, optimizer; kwargs...)
    return solve_model(file, model_type, optimizer, build_voltage_projection_opf; kwargs...)
end


function solve_power_projection_opf(file, model_type::Type, optimizer; kwargs...)
    return solve_model(file, model_type, optimizer, build_power_projection_opf; kwargs...)
end

function solve_power_ws_opf(file, model_type::Type, optimizer; kwargs...)
    return solve_model(file, model_type, optimizer, build_opf_ws; kwargs...)
end



function run_powermodels_opf_custom(json_path)
    println("--- PandaModels.jl: run_powermodels_opf called! ---")
    pm = _PdM.load_pm_from_json(json_path)
    active_powermodels_silence!(pm)
    pm = remove_extract_params!(pm)

    # NEW LOGIC HERE: Check for presence of target data
    power_projection = false
    voltage_projection = false
    power_ws = false
    voltage_ws = false

    # Warm start data Dict
    warm_start_data = Dict{String, Any}("bus" => Dict{String,Any}(), "gen" => Dict{String,Any}())

    # Check for presence of target_pg (which implies a power/PgVm projection setup)
    for (i, gen) in pm["gen"]
        if haskey(gen, "target_pg")
            power_projection = true
            break # No need to check other generators if one has target_pg
        end
    end

    # Check for presence of ws_pg (which implies a warm start PgVm setup)
    for (i, gen) in pm["gen"]
        if haskey(gen, "ws_pg")
            power_ws = true
            break # No need to check other generators if one has target_pg
        end
    end

    # Check for presence of target_va (which implies a voltage/VmVa projection setup)
    for (i, bus) in pm["bus"]
        if haskey(bus, "target_va")
            voltage_projection = true
            break # No need to check other buses if one has target_va
        end
    end

    # Check for presence of ws_va (which implies a voltage/VmVa ws setup)
    for (i, bus) in pm["bus"]
        if haskey(bus, "ws_va")
            voltage_ws = true
            break # No need to check other buses if one has target_va
        end
    end


    # populate warm start Dict
    """
    Warm start by adding _start
    https://lanl-ansi.github.io/PowerModels.jl/stable/power-flow/#:~:text=Warm%20Starting&text=In%20such%20a%20case%2C%20this,provide%20a%20suitable%20solution%20guess.
    """
    if power_ws # If any ws_pg was found, we assume this warm start scenario
        for (i, gen) in pm["gen"]
            if haskey(gen, "ws_pg")
                gen["pg_start"] = gen["ws_pg"]
                # gen["qg_start"] = get(gen, "ws_qg", 0.0) # Always include qg, default to 0.0 if not specified
            end
        end
        for (i, bus) in pm["bus"]
            if haskey(bus, "ws_vm")
                bus["vm_start"] = bus["ws_vm"]
                # bus["va_start"] = get(bus, "ws_va", 0.0)
            end
        end
    elseif voltage_ws # If only ws_va was found (and not ws_pg based on elseif logic)
        for (i, bus) in pm["bus"]
            if haskey(bus, "ws_vm") # We still need vm if it's available
                bus["vm_start"] = bus["ws_vm"]
                bus["va_start"] = deg2rad(bus["ws_va"])
            end
        end
    end


    result = nothing # Initialize result
    # run_projection = true # Set to true for now, will be checked later

    if power_projection # If target data is present, run your custom projection OPF
        println("--- Running Custom Power Projection OPF ---")
        model = get_model(pm["pm_model"]) # e.g., ACPPowerModel
        projection_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => pm["pm_log_level"])

        result = solve_power_projection_opf(
            pm,
            model,
            projection_solver, # Pass the newly defined projection_solver
            setting = Dict("output" => Dict("branch_flows" => true)),
        )

        print("\n power project solve time: ", result["solve_time"])
        print("\n power project objective: ", result["objective"])

    elseif voltage_projection # If target data is present, run your custom voltage projection OPF
        println("--- Running Custom Voltage Projection OPF ---")
        model = get_model(pm["pm_model"]) # e.g., ACPPowerModel
        projection_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => pm["pm_log_level"])

        result = solve_voltage_projection_opf(
            pm,
            model,
            projection_solver, # Pass the newly defined projection_solver
            setting = Dict("output" => Dict("branch_flows" => true)),
        )

        print("\n volt project solve time: ", result["solve_time"])
        print("\n volt project objective: ", result["objective"])


    elseif power_ws # If target data is present, run your custom voltage projection OPF
        println("--- Running Custom Power WS OPF ---")
        model = get_model(pm["pm_model"]) # e.g., ACPPowerModel
        projection_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => pm["pm_log_level"])

        result = solve_power_ws_opf(
            pm,
            model,
            projection_solver, # Pass the newly defined projection_solver
            setting = Dict("output" => Dict("branch_flows" => true)),
        )
        print("\n power ws solve time: ", result["solve_time"])
        print("\n power ws objective: ", result["objective"])
        
    elseif voltage_ws # If target data is present, run your custom voltage projection OPF
        println("--- Running Custom Voltage WS OPF ---")
        model = get_model(pm["pm_model"]) # e.g., ACPPowerModel
        projection_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => pm["pm_log_level"])

        result = solve_power_ws_opf(
            pm,
            model,
            projection_solver, # Pass the newly defined projection_solver
            setting = Dict("output" => Dict("branch_flows" => true)),
        )

        print("\n volt ws solve time: ", result["solve_time"])
        print("\n volt ws objective: ", result["objective"])

    else # Otherwise, run the standard OPF as before
        println("--- Running Normal OPF ---")
        cl = check_current_limit!(pm)
        if cl == 0
            pm = check_powermodels_data!(pm)

            solver = get_solver(pm) # The solver is needed here
            result = _PM.solve_opf(
                pm,
                get_model(pm["pm_model"]), # Get model again
                solver,
                setting = Dict("output" => Dict("branch_flows" => true)),
            )
        else
            solver = get_solver(pm) # The solver is needed here
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


function build_opf_ws(pm::AbstractPowerModel)
    variable_bus_voltage(pm)
    variable_gen_power(pm)
    variable_branch_power(pm)
    variable_dcline_power(pm)

    objective_min_fuel_and_flow_cost(pm)

    constraint_model_voltage(pm)

    for i in ids(pm, :ref_buses)
        constraint_theta_ref(pm, i)
    end

    for i in ids(pm, :bus)
        constraint_power_balance(pm, i)
    end

    for i in ids(pm, :branch)
        constraint_ohms_yt_from(pm, i)
        constraint_ohms_yt_to(pm, i)

        constraint_voltage_angle_difference(pm, i)

        constraint_thermal_limit_from(pm, i)
        constraint_thermal_limit_to(pm, i)
    end

    for i in ids(pm, :dcline)
        constraint_dcline_power_losses(pm, i)
    end
end



function build_power_projection_opf(pm::_PM.AbstractPowerModel)
    println("\n--- Starting _build_projection_opf ---")

    # Add extra variables for projection
    vars_collection = JuMP.VariableRef[]
    x_hat_values = Float64[]

    
    # Standard PowerModels variable declarations
    _PM.variable_bus_voltage(pm) # Will create :vm and :va for ACPPowerModel
    _PM.variable_gen_power(pm)   # Will create :pg and :qg
    _PM.variable_branch_power(pm)
    _PM.variable_dcline_power(pm, bounded = false)

    for (i, gen) in _PM.ref(pm, :gen)
        if haskey(gen, "target_pg")
            # Check if the :pg variable for this generator index 'i' exists in the JuMP model
            if haskey(_PM.var(pm), :pg) && i in _PM.ids(pm, :gen) && (i in axes(_PM.var(pm, :pg), 1))
                push!(vars_collection, _PM.var(pm, :pg, i))
                push!(x_hat_values, gen["target_pg"])
            else
                @warn "Generator $i has 'target_pg' but :pg variable not found for this index in PowerModels model. Skipping."
            end
        end
    end

    for (i, bus) in _PM.ref(pm, :bus)
        if haskey(bus, "target_vm")
            # Check if the :vm variable for this bus index 'i' exists in the JuMP model
            if haskey(_PM.var(pm), :vm) && i in _PM.ids(pm, :bus) && (i in axes(_PM.var(pm, :vm), 1))
                push!(vars_collection, _PM.var(pm, :vm, i))
                push!(x_hat_values, bus["target_vm"])
            else
                @warn "Bus $i has 'target_vm' but :vm variable not found for this index in PowerModels model. Skipping."
            end
        end
    end


    N = length(vars_collection)

    # Always define r and aux
    @variable(pm.model, r_var >= 0, base_name="r")
    pm.var[:r] = r_var # This assignment MUST happen before objective_projection is called

    if N > 0
        @variable(pm.model, aux_vars[1:N], base_name="aux")
        pm.var[:aux] = aux_vars # Store aux_vars only if N > 0

        for k in 1:N
            JuMP.@constraint(pm.model, aux_vars[k] == vars_collection[k])
        end
        JuMP.@constraint(pm.model, sum((aux_vars[k] - x_hat_values[k])^2 for k in 1:N) <= r_var^2)
    else
        @warn "No 'target_pg' or 'target_vm' found for active components. Projection constraint will not be added."
        JuMP.@constraint(pm.model, r_var == 0.0) # If N=0, r should be 0
    end

    pm.ext[:projection_vars] = vars_collection
    pm.ext[:projection_targets] = x_hat_values
    pm.ext[:projection_N] = N

    # Call the custom objective function. It will now find pm.var[:r].
    if N == 0
        @warn "No projection variables found. Setting objective to Min 0.0."
        JuMP.@objective(pm.model, Min, 0.0)
    else
        JuMP.@objective(pm.model, Min, r_var) # Use the local variable r_var directly
        # println("Objective set to Min r_var")
    end

    # Standard PowerModels constraint declarations
    _PM.constraint_model_voltage(pm)

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

    # println("Model is built!")
end




function build_voltage_projection_opf(pm::_PM.AbstractPowerModel)
    println("\n--- Starting build_voltage_projection_opf ---")

    # Initialize collections for projection variables and their targets
    vars_collection = JuMP.VariableRef[]
    x_hat_values = Float64[]

    # Declare standard PowerModels optimization variables.
    # These calls will populate pm.var[:vm], pm.var[:va], etc.
    _PM.variable_bus_voltage(pm) # Creates :vm and :va
    _PM.variable_gen_power(pm)   # Creates :pg and :qg (needed for power balance)
    _PM.variable_branch_power(pm)
    _PM.variable_dcline_power(pm, bounded = false)

    # Populate vars_collection and x_hat_values for BUS VOLTAGE MAGNITUDE (vm) targets
    for (i, bus) in _PM.ref(pm, :bus)
        if haskey(bus, "target_vm")
            if haskey(_PM.var(pm), :vm) && i in axes(_PM.var(pm, :vm), 1)
                push!(vars_collection, _PM.var(pm, :vm, i))
                push!(x_hat_values, bus["target_vm"])
            else
                @warn "Bus $i has 'target_vm' but :vm variable not found for this index in PowerModels model. Skipping."
            end
        end
    end

    # Populate vars_collection and x_hat_values for BUS VOLTAGE ANGLE (va) targets
    for (i, bus) in _PM.ref(pm, :bus)
        if haskey(bus, "target_va")
            if haskey(_PM.var(pm), :va) && i in axes(_PM.var(pm, :va), 1)
                push!(vars_collection, _PM.var(pm, :va, i))
                # Convert target_va from degrees to radians for JuMP's internal va variable (which is in radians)
                push!(x_hat_values, deg2rad(bus["target_va"]))
            else
                @warn "Bus $i has 'target_va' but :va variable not found for this index in PowerModels model. Skipping."
            end
        end
    end

    N = length(vars_collection)

    # Declare r_var and store it in pm.var[:r]
    @variable(pm.model, r_var >= 0, base_name="r")
    pm.var[:r] = r_var

    if N > 0
        @variable(pm.model, aux_vars[1:N], base_name="aux")
        pm.var[:aux] = aux_vars

        for k in 1:N
            JuMP.@constraint(pm.model, aux_vars[k] == vars_collection[k])
        end
        JuMP.@constraint(pm.model, sum((aux_vars[k] - x_hat_values[k])^2 for k in 1:N) <= r_var^2)
    else
        @warn "No 'target_vm' or 'target_va' found for active buses. Projection constraint will not be added."
        JuMP.@constraint(pm.model, r_var == 0.0)
    end

    pm.ext[:projection_vars] = vars_collection
    pm.ext[:projection_targets] = x_hat_values
    pm.ext[:projection_N] = N

    # Set the objective directly here
    println("--- Setting objective directly in build_voltage_projection_opf ---")
    if N == 0
        @warn "No projection variables found. Setting objective to Min 0.0."
        JuMP.@objective(pm.model, Min, 0.0)
    else
        JuMP.@objective(pm.model, Min, r_var)
        # println("Objective set to Min r_var")
    end

    # Standard PowerModels constraint declarations (essential for a valid OPF)
    _PM.constraint_model_voltage(pm)

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

    # println("Model is built!")
end












function run_powermodels_pf(json_path)
    pm = load_pm_from_json(json_path)
    active_powermodels_silence!(pm)
    pm = check_powermodels_data!(pm)
    # calculate branch power flows
    if pm["pm_model"] == "ACNative"
        result = _PM.compute_ac_pf(pm)
    elseif pm["pm_model"] == "DCNative"
        result = _PM.compute_dc_pf(pm)
    else
        model = get_model(pm["pm_model"])
        solver = get_solver(pm)
        result = _PM.solve_pf(
            pm,
            model,
            solver,
            setting = Dict("output" => Dict("branch_flows" => true)),
        )
    end

    # add result to net data
    _PM.update_data!(pm, result["solution"])
    # calculate branch power flows
    if pm["ac"]
        flows = _PM.calc_branch_flow_ac(pm)
    else
        flows = _PM.calc_branch_flow_dc(pm)
    end
    # add flow to net and result
    _PM.update_data!(result["solution"], flows)
    # _PM.update_data!(pm, result["solution"])
    # _PM.update_data!(pm, flows)
    return result
end

function run_powermodels_opf(json_path)
    pm = _PdM.load_pm_from_json(json_path)
    active_powermodels_silence!(pm)
    pm = remove_extract_params!(pm)
    model = get_model(pm["pm_model"])
    solver = get_solver(pm)

    cl = check_current_limit!(pm)

    if cl == 0
        pm = check_powermodels_data!(pm)
        result = _PM.solve_opf(
            pm,
            model,
            solver,
            setting = Dict("output" => Dict("branch_flows" => true)),
        )
    else

        # for (key, value) in pm["gen"]
        #    value["pmin"] /= pm["baseMVA"]
        #    value["pmax"] /= pm["baseMVA"]
        #    value["qmax"] /= pm["baseMVA"]
        #    value["qmin"] /= pm["baseMVA"]
        #    value["pg"] /= pm["baseMVA"]
        #    value["qg"] /= pm["baseMVA"]
        #    value["cost"] *= pm["baseMVA"]
        # end
        #
        # for (key, value) in pm["branch"]
        #    value["c_rating_a"] /= pm["baseMVA"]
        # end
        #
        # for (key, value) in pm["load"]
        #    value["pd"] /= pm["baseMVA"]
        #    value["qd"] /= pm["baseMVA"]
        # end

        result = _PM._solve_opf_cl(
            pm,
            model,
            solver,
            setting = Dict("output" => Dict("branch_flows" => true)),
        )
    end


    print(result) # Now print the potentially modified result

    return result
end

function run_powermodels_tnep(json_path)
    pm = _PdM.load_pm_from_json(json_path)
    active_powermodels_silence!(pm)
    pm = check_powermodels_data!(pm)
    pm = remove_extract_params!(pm)
    model = get_model(pm["pm_model"])
    solver = get_solver(pm)

    result = _PM.solve_tnep(
        pm,
        model,
        solver,
        setting = Dict("output" => Dict("branch_flows" => true)),
    )
    return result
end

function run_powermodels_ots(json_path)
    pm = _PdM.load_pm_from_json(json_path)
    active_powermodels_silence!(pm)
    pm = check_powermodels_data!(pm)
    pm = remove_extract_params!(pm)
    model = get_model(pm["pm_model"])
    solver = get_solver(pm)

    result = _PM.solve_ots(
        pm,
        model,
        solver,
        setting = Dict("output" => Dict("branch_flows" => true)),
    )
    return result
end

function run_powermodels_multi_storage(json_path)
    pm = _PdM.load_pm_from_json(json_path)
    active_powermodels_silence!(pm)
    pm = check_powermodels_data!(pm)
    model = get_model(pm["pm_model"])
    solver = get_solver(pm)
    mn = set_pq_values_from_timeseries(pm)

    result = _PM.solve_mn_opf_strg(mn, model, solver,
        setting = Dict("output" => Dict("branch_flows" => true)),
    )
    return result
end
