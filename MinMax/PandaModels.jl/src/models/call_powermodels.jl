using JuMP
using PowerModels
using Ipopt # Assuming Ipopt is your solver
using Juniper
using MathOptInterface # MOI
using InfrastructureModels # Needed for _guard_objective_value, _guard_objective_bound
using JSON
const _PM = PowerModels # Alias for convenience, as used in your example
import PandaModels as _PdM # Assuming this alias is already here
# using Gurobi
using Random
Random.seed!(1);

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

function solve_opf(file, model_type::Type, optimizer; kwargs...)
    return solve_model(file, model_type, optimizer, build_opf; kwargs...)
end



function run_powermodels_opf_custom(json_path)
    println("--- PandaModels.jl: run_powermodels_opf called! ---")

    # 1. Create an in-memory IOBuffer from the string.
    #io_buffer = IOBuffer(json_path)
    # 2. Parse the JSON data from this in-memory buffer, not from a file.
    pm = PowerModels.parse_json(json_path)

    ######
    tight_angmin = -pi/6   # -0.5235987755982988
    tight_angmax =  pi/6   # 0.5235987755982988

    for (key, br) in pm["branch"]
        # Set angle limits
        br["angmin"] = tight_angmin
        br["angmax"] = tight_angmax

        # Fix ratings for both lines and transformers
        br["rate_a"] /= 100
        br["rate_b"] = br["rate_a"]#/= 100
        br["rate_c"] = br["rate_a"]#/= 100

        Vbase = pm["bus"][string(br["f_bus"])]["base_kv"]
        Sbase = 100.0  # or whatever system base MVA you use
        Zbase = (Vbase^2) / Sbase

        # Scale R/X depending on branch type
        if !br["transformer"]
            
            # Lines: scale R/X normally
            br["br_r"] *= Sbase
            br["br_x"] *= Sbase

            # Scale line charging / shunts
            br["b_fr"] /= Sbase
            br["b_to"] /= Sbase
            # br["g_fr"] /= Sbase
            # br["g_to"] /= Sbase
        else
            # print(br)
            # print("\n")
            # Lines: scale R/X normally
            br["br_r"] *= Sbase
            br["br_x"] *= Sbase

            # Scale line charging / shunts
            br["b_fr"] /= Sbase
            br["b_to"] /= Sbase
            # br["g_fr"] /= Sbase
            # br["g_to"] /= Sbase
            #continue
            # br["shift"] = deg2rad(br["shift"])  # only if imported in degrees
        end
 
    end

    # if haskey(pm, "shunt")
    #     for (key, sh) in pm["shunt"] # shunts are already in pu!
    #         # print(sh)
    #         # print("\n")
    #         sh["gs"] /= 100
    #         sh["bs"] /= 100
    #         #print("SHUUUNT")
    #     end
    # end

    for (key, gen) in pm["gen"]
        
        if !haskey(gen, "qg") || gen["qg"] == 0.0
            gen["qg"] = (gen["qmax"] + gen["qmin"]) / 2  # midpoint heuristic
        end

        if !haskey(gen, "pg") || gen["pg"] == 0.0
            gen["pg"] = (gen["pmax"] + gen["pmin"]) / 2  # midpoint heuristic
        end

        # scale ratings
        gen["pg"]   /= 100
        gen["qg"]   /= 100
        gen["pmax"] /= 100
        gen["pmin"] /= 100
        gen["qmax"] /= 100
        gen["qmin"] /= 100

        # fix tiny numerical noise (absolute value < 1e-4)
        for field in ["pg", "qg", "pmax", "pmin", "qmax", "qmin"]
            if abs(gen[field]) < 1e-4
                gen[field] = 0.0
            end
        end

        # rescale cost coefficients
        if haskey(gen, "cost")
            cost = gen["cost"]
            if length(cost) == 3
                # quadratic cost: a*p^2 + b*p + c
                gen["cost"] = [cost[1] * 100^2, cost[2] * 100, cost[3]]
            elseif length(cost) == 2
                # linear cost: a*p + b
                gen["cost"] = [cost[1] * 100, cost[2]]
            end
        end
    end

    for (key, bus) in pm["bus"]
        if bus["bus_type"] == 3
            bus["vmin"] = 0.9999
            bus["vmax"] = 1.0001
        else
            bus["vmin"] = min(bus["vmin"], 0.94)
            bus["vmax"] = max(bus["vmax"], 1.06)
        end
    end

    for (key, load) in pm["load"]
        # Example: scale load by 1/100
        load["pd"] /= 100
        load["qd"] /= 100
    end


    #pm = _PdM.load_pm_from_json(json_path)
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


    # # define solver for all cases:
    # solver = JuMP.optimizer_with_attributes(
    #             Ipopt.Optimizer,
    #             "print_level" => 3,
    #             "max_iter" => 3000,
    #             "tol" => 1e-8,
    #             "dual_inf_tol" => 1e-6,   # dual infeasibility
    #             "compl_inf_tol" => 1e-6,  # complementarity tolerance
    #             "constr_viol_tol" => 1e-4, # default: 1e-4
    #             "acceptable_tol" => 1e-3,
    #             "hessian_approximation" => "exact",
    #             # "bound_push" => 5e-1, # default: 1e-2, ensures initial point is strictly feasible
    #             # "slack_bound_push" => 5e-1, # default: 1e-2 
    #             # "least_square_init_primal" => "yes",
    #             # "least_square_init_duals" => "yes",
    #             # "acceptable_dual_inf_tol" => 1e6,
    #             # "accept_every_trial_step" => "yes",
    #             # "nlp_scaling_method" => "none", 
    #             # "mu_strategy" => "adaptive",
    #             # "mu_init" => 1e-3,
    #             # "linear_solver" => "ma27",   
    #             # "mehrotra_algorithm" => "yes",
    #         )

    solver = JuMP.optimizer_with_attributes(
                Ipopt.Optimizer,
                "print_level" => 0,
                "max_cpu_time" => 10.0,
                #"linear_solver" => "ma27",   
                )


    result = nothing # Initialize result
    # run_projection = true # Set to true for now, will be checked later

    if power_projection # If target data is present, run your custom projection OPF
        println("--- Running Custom Power Projection OPF ---")
        model = get_model(pm["pm_model"]) # e.g., ACPPowerModel
        # print(model["load"])
        # print(model["gen"])
        projection_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => pm["pm_log_level"])
        pm = check_powermodels_data!(pm)

        result = solve_power_projection_opf(
            pm,
            model,
            solver, # projection_solver, # Pass the newly defined projection_solver
            setting = Dict("output" => Dict("branch_flows" => true)),
        )

        # print("\n power project solve time: ", result["solve_time"])
        # print("\n power project objective: ", result["objective"])

    elseif voltage_projection # If target data is present, run your custom voltage projection OPF
        println("--- Running Custom Voltage Projection OPF ---")
        model = get_model(pm["pm_model"]) # e.g., ACPPowerModel
        projection_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => pm["pm_log_level"])
        pm = check_powermodels_data!(pm)

        result = solve_voltage_projection_opf(
            pm,
            model,
            solver, #projection_solver, # Pass the newly defined projection_solver
            setting = Dict("output" => Dict("branch_flows" => true)),
        )

        # print("\n volt project solve time: ", result["solve_time"])
        # print("\n volt project objective: ", result["objective"])


    elseif power_ws # If target data is present, run your custom voltage projection OPF
        println("--- Running Custom Power WS OPF ---")
        model = get_model(pm["pm_model"]) # e.g., ACPPowerModel
        projection_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => pm["pm_log_level"])
        pm = check_powermodels_data!(pm)

        result = solve_opf(
            pm,
            model,
            solver, #projection_solver, # Pass the newly defined projection_solver
            setting = Dict("output" => Dict("branch_flows" => true)),
        )
        # print("\n power ws solve time: ", result["solve_time"])
        # print("\n power ws objective: ", result["objective"])
        
    elseif voltage_ws # If target data is present, run your custom voltage projection OPF
        println("--- Running Custom Voltage WS OPF ---")
        model = get_model(pm["pm_model"]) # e.g., ACPPowerModel
        projection_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => pm["pm_log_level"])
        pm = check_powermodels_data!(pm)

        result = solve_opf(
            pm,
            model,
            solver, #projection_solver, # Pass the newly defined projection_solver
            setting = Dict("output" => Dict("branch_flows" => true)),
        )

        # print("\n volt ws solve time: ", result["solve_time"])
        # print("\n volt ws objective: ", result["objective"])

    else # Otherwise, run the standard OPF as before
        println("--- Running Normal OPF ---")
        cl = check_current_limit!(pm)
        if cl == 0
            pm = check_powermodels_data!(pm)

            #solver = get_solver(pm) # The solver is needed here
            result = solve_opf(
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

    # solution = result["solution"]
    # rescale_pm_results_back!(solution; Sbase = 100.0)
    # result["solution"] = solution

    return result
end



function rescale_pm_results_back!(pm::Dict{String, Any}; Sbase::Float64 = 100.0)
    # --- Branches ---
    # for (key, br) in pm["branch"]
    #     # Multiply by Sbase to get original R/X
    #     br["br_r"] /= Sbase
    #     br["br_x"] /= Sbase

    #     # Multiply shunt admittance by Sbase to go back
    #     br["b_fr"] *= Sbase
    #     br["b_to"] *= Sbase
    #     if haskey(br, "g_fr")
    #         br["g_fr"] *= Sbase
    #         br["g_to"] *= Sbase
    #     end

    #     # Multiply ratings back
    #     if haskey(br, "rate_a")
    #         br["rate_a"] *= Sbase
    #         br["rate_b"] = br["rate_a"]
    #         br["rate_c"] = br["rate_a"]
    #     end
    # end

    # --- Generators ---
    for (key, gen) in pm["gen"]
        for field in ["pg", "qg"]# "pmax", "pmin", "qmax", "qmin"]
            if haskey(gen, field)
                gen[field] *= Sbase
            end
        end

        # # Rescale costs back
        # if haskey(gen, "cost")
        #     cost = gen["cost"]
        #     if length(cost) == 3
        #         gen["cost"] = [cost[1] / Sbase^2, cost[2] / Sbase, cost[3]]
        #     elseif length(cost) == 2
        #         gen["cost"] = [cost[1] / Sbase, cost[2]]
        #     end
        # end
    end

    # --- Loads ---
    # for (key, load) in pm["load"]
    #     for field in ["pd", "qd"]
    #         if haskey(load, field)
    #             load[field] *= Sbase
    #         end
    #     end
    # end

    # # --- Shunts ---
    # if haskey(pm, "shunt")
    #     for (key, sh) in pm["shunt"]
    #         for field in ["gs", "bs"]
    #             if haskey(sh, field)
    #                 sh[field] *= Sbase
    #             end
    #         end
    #     end
    # end

    return pm
end



function build_opf(pm::AbstractPowerModel)
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



# function build_power_projection_opf(pm::_PM.AbstractPowerModel)
#     println("\n--- Starting _build_projection_opf ---")

#     # Add extra variables for projection
#     vars_collection = JuMP.VariableRef[]
#     x_hat_values = Float64[]

    
#     # Standard PowerModels variable declarations
#     _PM.variable_bus_voltage(pm) # Will create :vm and :va for ACPPowerModel
#     _PM.variable_gen_power(pm)   # Will create :pg and :qg
#     _PM.variable_branch_power(pm)
#     _PM.variable_dcline_power(pm, bounded = false)

#     for (i, gen) in _PM.ref(pm, :gen)
#         if haskey(gen, "target_pg")
#             # Check if the :pg variable for this generator index 'i' exists in the JuMP model
#             if haskey(_PM.var(pm), :pg) && i in _PM.ids(pm, :gen) && (i in axes(_PM.var(pm, :pg), 1))
#                 push!(vars_collection, _PM.var(pm, :pg, i))
#                 push!(x_hat_values, gen["target_pg"])
#             else
#                 @warn "Generator $i has 'target_pg' but :pg variable not found for this index in PowerModels model. Skipping."
#             end
#         end
#     end

#     for (i, bus) in _PM.ref(pm, :bus)
#         if haskey(bus, "target_vm")
#             # Check if the :vm variable for this bus index 'i' exists in the JuMP model
#             if haskey(_PM.var(pm), :vm) && i in _PM.ids(pm, :bus) && (i in axes(_PM.var(pm, :vm), 1))
#                 push!(vars_collection, _PM.var(pm, :vm, i))
#                 push!(x_hat_values, bus["target_vm"])
#             else
#                 @warn "Bus $i has 'target_vm' but :vm variable not found for this index in PowerModels model. Skipping."
#             end
#         end
#     end


#     N = length(vars_collection)

#     # Always define r and aux
#     @variable(pm.model, r_var >= 0, base_name="r")
#     pm.var[:r] = r_var # This assignment MUST happen before objective_projection is called

#     # if N > 0
#     #     @variable(pm.model, aux_vars[1:N], base_name="aux")
#     #     pm.var[:aux] = aux_vars # Store aux_vars only if N > 0

#     #     for k in 1:N
#     #         JuMP.@constraint(pm.model, aux_vars[k] == vars_collection[k])
#     #     end
#     #     JuMP.@constraint(pm.model, sum((aux_vars[k] - x_hat_values[k])^2 for k in 1:N) <= r_var^2)

#     # else
#     #     @warn "No 'target_pg' or 'target_vm' found for active components. Projection constraint will not be added."
#     #     JuMP.@constraint(pm.model, r_var == 0.0) # If N=0, r should be 0
#     # end

#     if N == 0
#         @warn "No 'target_pg' or 'target_vm' found. Projection constraint skipped."
#         @constraint(pm.model, r_var == 0)
#         @objective(pm.model, Min, 0)
#     else
#         # Direct quadratic constraint, no aux vars
#         @constraint(pm.model, sum((vars_collection .- x_hat_values).^2) <= r_var^2)
#         @objective(pm.model, Min, r_var)
#     end

#     pm.ext[:projection_vars] = vars_collection
#     pm.ext[:projection_targets] = x_hat_values
#     pm.ext[:projection_N] = N

#     # Call the custom objective function. It will now find pm.var[:r].
#     if N == 0
#         @warn "No projection variables found. Setting objective to Min 0.0."
#         JuMP.@objective(pm.model, Min, 0.0)
#     else
#         JuMP.@objective(pm.model, Min, r_var) # Use the local variable r_var directly
#         # println("Objective set to Min r_var")
#     end

#     # Standard PowerModels constraint declarations
#     _PM.constraint_model_voltage(pm)

#     for i in _PM.ids(pm, :ref_buses)
#         _PM.constraint_theta_ref(pm, i)
#     end

#     for i in _PM.ids(pm, :bus)
#         _PM.constraint_power_balance(pm, i)
#     end

#     for (i, branch) in _PM.ref(pm, :branch)
#         _PM.constraint_ohms_yt_from(pm, i)
#         _PM.constraint_ohms_yt_to(pm, i)
#         _PM.constraint_thermal_limit_from(pm, i)
#         _PM.constraint_thermal_limit_to(pm, i)
#     end

#     for i in _PM.ids(pm, :dcline)
#         _PM.constraint_dcline_power_losses(pm, i)
#     end

#     # println("Model is built!")
# end


function build_power_projection_opf(pm::_PM.AbstractPowerModel)
    println("\n--- Starting _build_projection_opf ---")

    # === Step 1: Standard variable declarations ===
    _PM.variable_bus_voltage(pm)
    _PM.variable_gen_power(pm)
    _PM.variable_branch_power(pm)
    _PM.variable_dcline_power(pm)

    gen_ref = _PM.ref(pm, :gen)
    bus_ref = _PM.ref(pm, :bus)
    gen_vars = get(_PM.var(pm), :pg, Dict())
    bus_vars = get(_PM.var(pm), :vm, Dict())

    # === Step 2-5: Collect generator and bus targets ===
    vars_collection = JuMP.VariableRef[]
    x_hat_values = Float64[]

    # Generator targets
    for (i, gen) in gen_ref
        if haskey(gen, "target_pg") && i in axes(gen_vars, 1)
            push!(vars_collection, gen_vars[i])
            push!(x_hat_values, gen["target_pg"])
            JuMP.set_start_value(gen_vars[i], gen["target_pg"])  # warm start
        end
    end

    # Bus voltage magnitude targets
    for (i, bus) in bus_ref
        if haskey(bus, "target_vm") && i in axes(bus_vars, 1)
            push!(vars_collection, bus_vars[i])
            push!(x_hat_values, bus["target_vm"])
            JuMP.set_start_value(bus_vars[i], bus["target_vm"])  # warm start
        end
    end

    # Number of projection variables
    N = length(vars_collection)


    # === Step 3: Projection constraint & objective ===
    # @variable(pm.model, r_var >= 0)
    # pm.var[:r] = r_var

    # if N == 0
    #     @warn "No 'target_pg' or 'target_vm' found. Skipping projection constraint."
    #     @constraint(pm.model, r_var == 0)
    #     @objective(pm.model, Min, 0)
    #     return
    # end

    # ### quadratic
    # @constraint(pm.model, sum((vars_collection .- x_hat_values).^2) <= r_var^2)
    # @objective(pm.model, Min, r_var)

    #### least squares
    # weights = [1.0 / max(abs(x_hat_values[i]), 1e-3) for i in 1:N]
    @objective(pm.model, Min, sum((vars_collection .- x_hat_values).^2)) # weights[i] *

    #### inifinity norm
    # r_var_tmp = @variable(pm.model, base_name = "", lower_bound = 0)
    # for i in 1:N
    #     @constraint(pm.model, vars_collection[i] - x_hat_values[i] <= r_var_tmp)
    #     @constraint(pm.model, x_hat_values[i] - vars_collection[i] <= r_var_tmp)
    # end
    # @objective(pm.model, Min, r_var_tmp)

    # === Step 4: Power flow constraints ===
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
        _PM.constraint_voltage_angle_difference(pm, i)
        _PM.constraint_thermal_limit_from(pm, i)
        _PM.constraint_thermal_limit_to(pm, i)
    end
    for i in _PM.ids(pm, :dcline)
        _PM.constraint_dcline_power_losses(pm, i)
    end
end


function build_voltage_projection_opf(pm::_PM.AbstractPowerModel)
    println("\n--- Starting build_voltage_projection_opf (optimized) ---")

    # === Step 1: Declare standard PowerModels variables ===
    _PM.variable_bus_voltage(pm)  # creates :vm and :va
    _PM.variable_gen_power(pm)    # needed for power balance (:pg and :qg)
    _PM.variable_branch_power(pm)
    _PM.variable_dcline_power(pm)

    # get slack bus
    ref_bus_indices = Set(_PM.ids(pm, :ref_buses))

    # === Step 2: References and valid indices ===
    bus_ref = _PM.ref(pm, :bus)
    vm_vars = get(_PM.var(pm), :vm, Dict())
    va_vars = get(_PM.var(pm), :va, Dict())

    vars_collection = JuMP.VariableRef[]
    x_hat_values = Float64[]

    # === Voltage magnitude targets ===
    for (i, bus) in bus_ref
        # Skip slack/reference buses
        if i in ref_bus_indices
            continue
        end
        if haskey(bus, "target_vm") && i in axes(vm_vars, 1)
            push!(vars_collection, vm_vars[i])
            push!(x_hat_values, bus["target_vm"])
            JuMP.set_start_value(vm_vars[i], bus["target_vm"])

            push!(vars_collection, va_vars[i])
            push!(x_hat_values, deg2rad(bus["target_va"]))
            JuMP.set_start_value(va_vars[i], deg2rad(bus["target_va"]))
        end

        # if haskey(bus, "target_va") && i in axes(va_vars, 1)
        #     push!(vars_collection, va_vars[i])
        #     push!(x_hat_values, deg2rad(bus["target_va"]))
        #     JuMP.set_start_value(va_vars[i], deg2rad(bus["target_va"]))
        # end
    end

    N = length(vars_collection)

    # === Step 6: Projection variable & quadratic constraint ===
    # @variable(pm.model, r_var >= 0)
    # pm.var[:r] = r_var

    # #### Direct quadratic constraint (SOCP)
    # @constraint(pm.model, sum((vars_collection .- x_hat_values).^2) <= r_var^2)
    # @objective(pm.model, Min, r_var)

    #### least squares
    # weights = [1.0 / max(abs(x_hat_values[i]), 1e-3) for i in 1:N]
    @objective(pm.model, Min, sum((vars_collection .- x_hat_values).^2)) # weights[i] * 

    # if N > 0
    #     #### Direct quadratic constraint (no aux_vars needed)
    #     @constraint(pm.model, sum((vars_collection .- x_hat_values).^2) <= r_var^2)
    #     @objective(pm.model, Min, r_var)

    #     #### least squares
    #     # weights = [1.0 / max(abs(x_hat_values[i]), 1e-3) for i in 1:N]
    #     # @objective(pm.model, Min, sum((vars_collection .- x_hat_values).^2)) # weights[i] * 

    #     #### inifinity norm
    #     # r_var_tmp = @variable(pm.model, base_name = "", lower_bound = 0)
    #     # for i in 1:N
    #     #     @constraint(pm.model, vars_collection[i] - x_hat_values[i] <= r_var_tmp)
    #     #     @constraint(pm.model, x_hat_values[i] - vars_collection[i] <= r_var_tmp)
    #     # end
    #     # @objective(pm.model, Min, r_var_tmp)

    # else
    #     @warn "No 'target_vm' or 'target_va' found. Projection constraint skipped."
    #     # @constraint(pm.model, r_var == 0)
    #     @objective(pm.model, Min, 0)
    # end

    # Store projection info in pm.ext
    pm.ext[:projection_vars] = vars_collection
    pm.ext[:projection_targets] = x_hat_values
    pm.ext[:projection_N] = N

    # === Step 7: Standard PowerModels constraints ===
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
        _PM.constraint_voltage_angle_difference(pm, i)
        _PM.constraint_thermal_limit_from(pm, i)
        _PM.constraint_thermal_limit_to(pm, i)
    end

    for i in _PM.ids(pm, :dcline)
        _PM.constraint_dcline_power_losses(pm, i)
    end

    # println("--- Voltage projection OPF model built ---")
end






# function build_voltage_projection_opf(pm::_PM.AbstractPowerModel)
#     println("\n--- Starting build_voltage_projection_opf ---")

#     # Initialize collections for projection variables and their targets
#     vars_collection = JuMP.VariableRef[]
#     x_hat_values = Float64[]

#     # Declare standard PowerModels optimization variables.
#     # These calls will populate pm.var[:vm], pm.var[:va], etc.
#     _PM.variable_bus_voltage(pm) # Creates :vm and :va
#     _PM.variable_gen_power(pm)   # Creates :pg and :qg (needed for power balance)
#     _PM.variable_branch_power(pm)
#     _PM.variable_dcline_power(pm, bounded = false)

#     # Populate vars_collection and x_hat_values for BUS VOLTAGE MAGNITUDE (vm) targets
#     for (i, bus) in _PM.ref(pm, :bus)
#         if haskey(bus, "target_vm")
#             if haskey(_PM.var(pm), :vm) && i in axes(_PM.var(pm, :vm), 1)
#                 push!(vars_collection, _PM.var(pm, :vm, i))
#                 push!(x_hat_values, bus["target_vm"])
#             else
#                 @warn "Bus $i has 'target_vm' but :vm variable not found for this index in PowerModels model. Skipping."
#             end
#         end
#     end

#     # Populate vars_collection and x_hat_values for BUS VOLTAGE ANGLE (va) targets
#     for (i, bus) in _PM.ref(pm, :bus)
#         if haskey(bus, "target_va")
#             if haskey(_PM.var(pm), :va) && i in axes(_PM.var(pm, :va), 1)
#                 push!(vars_collection, _PM.var(pm, :va, i))
#                 # Convert target_va from degrees to radians for JuMP's internal va variable (which is in radians)
#                 push!(x_hat_values, deg2rad(bus["target_va"]))
#             else
#                 @warn "Bus $i has 'target_va' but :va variable not found for this index in PowerModels model. Skipping."
#             end
#         end
#     end

#     N = length(vars_collection)

#     # Declare r_var and store it in pm.var[:r]
#     @variable(pm.model, r_var >= 0, base_name="r")
#     pm.var[:r] = r_var

#     if N > 0
#         @variable(pm.model, aux_vars[1:N], base_name="aux")
#         pm.var[:aux] = aux_vars

#         for k in 1:N
#             JuMP.@constraint(pm.model, aux_vars[k] == vars_collection[k])
#         end
#         JuMP.@constraint(pm.model, sum((aux_vars[k] - x_hat_values[k])^2 for k in 1:N) <= r_var^2)
#     else
#         @warn "No 'target_vm' or 'target_va' found for active buses. Projection constraint will not be added."
#         JuMP.@constraint(pm.model, r_var == 0.0)
#     end

#     pm.ext[:projection_vars] = vars_collection
#     pm.ext[:projection_targets] = x_hat_values
#     pm.ext[:projection_N] = N

#     # Set the objective directly here
#     println("--- Setting objective directly in build_voltage_projection_opf ---")
#     if N == 0
#         @warn "No projection variables found. Setting objective to Min 0.0."
#         JuMP.@objective(pm.model, Min, 0.0)
#     else
#         JuMP.@objective(pm.model, Min, r_var)
#         # println("Objective set to Min r_var")
#     end

#     # Standard PowerModels constraint declarations (essential for a valid OPF)
#     _PM.constraint_model_voltage(pm)

#     for i in _PM.ids(pm, :ref_buses)
#         _PM.constraint_theta_ref(pm, i)
#     end

#     for i in _PM.ids(pm, :bus)
#         _PM.constraint_power_balance(pm, i)
#     end

#     for (i, branch) in _PM.ref(pm, :branch)
#         _PM.constraint_ohms_yt_from(pm, i)
#         _PM.constraint_ohms_yt_to(pm, i)
#         _PM.constraint_thermal_limit_from(pm, i)
#         _PM.constraint_thermal_limit_to(pm, i)
#     end

#     for i in _PM.ids(pm, :dcline)
#         _PM.constraint_dcline_power_losses(pm, i)
#     end

#     # println("Model is built!")
# end












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
    
    ### custom
    # 1. Create an in-memory IOBuffer from the string.
    io_buffer = IOBuffer(json_path)
    # 2. Parse the JSON data from this in-memory buffer, not from a file.
    pm = PowerModels.parse_json(io_buffer)

    ### original
    #pm = _PdM.load_pm_from_json(json_path)


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


    # print(result) # Now print the potentially modified result

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
