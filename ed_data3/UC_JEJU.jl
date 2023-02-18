using JuMP, HiGHS
using Plots; plotly();
using VegaLite  # to make some nice plots
using DataFrames, CSV, PrettyTables
using FileIO
ENV["COLUMNS"]=120; # Set so all columns of DataFrames and Matrices are displayed

datadir = joinpath("ed_data")

gen_info = CSV.read(joinpath(datadir,"Generators_data.csv"), DataFrame);
fuels = CSV.read(joinpath(datadir,"Fuels_data.csv"), DataFrame);
loads = CSV.read(joinpath(datadir,"Demand_20220701.csv"), DataFrame);
gen_variable = CSV.read(joinpath(datadir,"Generators_variability.csv"), DataFrame);

for f in [gen_info, fuels, loads, gen_variable]
    rename!(f,lowercase.(names(f)))
end

select!(gen_info, 1:26, :stor)
gen_df = outerjoin(gen_info,  fuels, on = :fuel) # load in fuel costs and add to data frame
rename!(gen_df, :cost_per_mmbtu => :fuel_cost)
gen_df[ismissing.(gen_df[:,:fuel_cost]), :fuel_cost] .= 0
gen_df[!, :is_variable] .= false 
gen_df[in(["onshore_wind_turbine","small_hydroelectric","solar_photovoltaic"]).(gen_df.resource), :is_variable] .= true;

gen_df.gen_full = lowercase.(gen_df.region .* "_" .* gen_df.resource .* "_" .* string.(gen_df.cluster) .* ".0");
gen_df = gen_df[gen_df.existing_cap_mw .> 0,:];
gen_variable.hour = mod.(gen_variable.hour .- 9, 8760) .+1 
sort!(gen_variable, :hour)
loads.hour = mod.(loads.hour .- 9, 8760) .+ 1
sort!(loads, :hour);
gen_variable     
describe(gen_variable)
gen_variable_long = stack(gen_variable, 
                        Not(:hour), 
                        variable_name=:gen_full,
                        value_name=:cf);

function value_to_df_puvw(var1, var2, var3, var4)
    solution1 = DataFrame(var1.data, :auto)
    ax1 = var1.axes[1]
    ax2 = var1.axes[2]
    cols = names(solution1)
    insertcols!(solution1, 1, :r_id => ax1)
    solution1 = stack(solution1, Not(:r_id), variable_name=:hour)
    solution1.hour = foldl(replace, [cols[i] => ax2[i] for i in 1:length(ax2)], init=solution1.hour)
    rename!(solution1, :value => :gen)
    solution1.hour = convert.(Int64,solution1.hour)
    solution2 = DataFrame(var2.data, :auto)
    ax1 = var2.axes[1]
    ax2 = var2.axes[2]
    cols = names(solution2)
    insertcols!(solution2, 1, :r_id => ax1)
    solution2 = stack(solution2, Not(:r_id), variable_name=:hour)
    solution2.hour = foldl(replace, [cols[i] => ax2[i] for i in 1:length(ax2)], init=solution2.hour)
    rename!(solution2, :value => :u_binary)
    solution2.hour = convert.(Int64,solution2.hour)
    solution3 = DataFrame(var3.data, :auto)
    ax1 = var3.axes[1]
    ax2 = var3.axes[2]
    cols = names(solution3)
    insertcols!(solution3, 1, :r_id => ax1)
    solution3 = stack(solution3, Not(:r_id), variable_name=:hour)
    solution3.hour = foldl(replace, [cols[i] => ax2[i] for i in 1:length(ax2)], init=solution3.hour)
    rename!(solution3, :value => :v_binary)
    solution3.hour = convert.(Int64,solution3.hour)
    solution4 = DataFrame(var4.data, :auto)
    ax1 = var4.axes[1]
    ax2 = var4.axes[2]
    cols = names(solution4)
    insertcols!(solution4, 1, :r_id => ax1)
    solution4 = stack(solution4, Not(:r_id), variable_name=:hour)
    solution4.hour = foldl(replace, [cols[i] => ax2[i] for i in 1:length(ax2)], init=solution4.hour)
    rename!(solution4, :value => :w_binary)
    solution4.hour = convert.(Int64,solution4.hour)
    return solution1, solution2, solution3, solution4
end



function unit_commitment_multi_time(gen_df, loads_multi, gen_variable_multi, mip_gap)
UC = Model(HiGHS.Optimizer)
set_optimizer_attribute(UC, "mip_rel_gap", mip_gap)

# Define sets based on data
# Thermal resources for which unit commitment constraints apply
G_thermal = gen_df[gen_df[!,:up_time] .> 0,:r_id] 
# Non-thermal resources for which unit commitment constraints do NOT apply 
G_nonthermal = gen_df[gen_df[!,:up_time] .== 0,:r_id]
G_var = gen_df[gen_df[!,:is_variable] .== 1,:r_id]
G_nonvar = gen_df[gen_df[!,:is_variable] .== 0,:r_id]
# Non-variable and non-thermal resources
G_nt_nonvar = intersect(G_nonvar, G_nonthermal)
G = gen_df.r_id
T = loads_multi.hour
T_red = loads_multi.hour[1:end-1]  # time periods used for ramp constraints

# Generator capacity factor time series for variable generators
gen_var_cf = innerjoin(gen_variable_multi, 
                gen_df[gen_df.is_variable .== 1 , 
                    [:r_id, :gen_full, :existing_cap_mw]], 
                on = :gen_full)
    
# Decision variables   
@variables(UC, begin
    p1[G_thermal, T]  >= 0  
    p2[G_thermal, T]  >= 0 
    p[G, T] >= 0
    u1[G_thermal, T], Bin 
    u2[G_thermal, T], Bin 
    u[G_thermal, T] >= 0
    # ED를 UC로 변경하기 위해 운영여부 u[g,t]변수 추가(Binary)
    v[G_thermal, T], Bin
    # 정지계획 추가를 위해 정지(Shut-dowm)여부 w[g,t]변수 추가(Binary)
    w[G_thermal, T], Bin
 end)
 
 # Objective function
@objective(UC, Min, 
    sum((gen_df[gen_df.r_id .== i,:heat_rate_mmbtu_per_mwh][1] * gen_df[gen_df.r_id .== i,:fuel_cost][1] +
        gen_df[gen_df.r_id .== i,:var_om_cost_per_mwh][1]) * p[i,t] 
                    for i in G_nonvar for t in T) + 
    sum(gen_df[gen_df.r_id .== i,:var_om_cost_per_mwh][1] * p[i,t] 
                    for i in G_var for t in T) +
    sum(gen_df[gen_df.r_id .== i,:start_cost_per_mw][1] * gen_df[gen_df.r_id .== i,:existing_cap_mw][1] * v[i,t] 
                    for i in G_thermal for t in T) +
    sum(500*u[i,t] for i in G_thermal for t in T) #기동고정비 항목이나 현재 그냥 500 추후 수정
)



# u와 p의 각각 정의
@constraint(UC, Total_p[i in G_thermal, t in T], 
            p[i,t] == p1[i,t] + p2[i,t])
@constraint(UC, Total_u[i in G_thermal, t in T], 
            u[i,t] == u1[i,t] + u2[i,t])

# u와 제약 설정
@constraint(UC, ucon[i in G_thermal, t in T], 
            1 >= u1[i,t] + u2[i,t])


#= output 변수 관련 제약: QP 문제가 되어 풀이가 안 되므로 p1, p2로 구간을 나누어 선형으로 접근하는 방식을 대신 이용
@constraint(UC, ou[i in G_thermal, t in T], 
            o[i,t] <= u[i,t])
@constraint(UC, o1[i in G_thermal, t in T], 
            p[i,t] * o[i,t] >= gen_df[gen_df.r_id .== i,:existing_cap_mw][1] *    
            gen_df[gen_df.r_id .== i,:min_power][1] * o[i,t])
@constraint(UC, o2[i in G_thermal, t in T], 
            p[i,t] * o[i,t] >= p[i,t] - gen_df[gen_df.r_id .== i,:existing_cap_mw][1] *
            gen_df[gen_df.r_id .== i,:min_power][1])=#
            
# u, v, w 관계 제약식
@constraint(UC, Binary_Formulations[i in G_thermal, t in T_red], 
           u[i,t+1] - u[i,t] == v[i,t+1] - w[i,t+1])

# 동일 시간에 v, w가 동시에 1 불가      
@constraint(UC, vw[i in G_thermal, t in T], 
            v[i,t] + w[i,t] <= 1)

#=
 intersection 연산이 느린 것 같아 다른 것으로 일단 대체
# Minimum Up Time Constraints
@constraint(UC, MUT[i in G_thermal, t in T],
u[i, t] >= sum(v[i, tt] 
                for tt in intersect(T,
                    (t-gen_df[gen_df.r_id .== i,:up_time][1]):t)))
# Minimum Down Time Constraints           
@constraint(UC, MDT[i in G_thermal, t in T],
1-u[i, t] >= sum(w[i, tt] 
                for tt in intersect(T,
                    (t-gen_df[gen_df.r_id .== i,:down_time][1]):t)))
=#

T_MUT(i) = T[1:end-1-gen_df[gen_df.r_id .== i,:up_time][1]]
T_MDT(i) = T[1:end-1-gen_df[gen_df.r_id .== i,:down_time][1]]

# Minimum Up Time Constraints
@constraint(UC, MUT[i in G_thermal, t in T_MUT(i)], 
            sum(u[i,t+1] for t = t:t+gen_df[gen_df.r_id .== i,:up_time][1]) >= (u[i,t+1] - u[i,t])*min(gen_df[gen_df.r_id .== i,:up_time][1],(T_period[end]-t-1)))

# Minimum Down Time Constraints           
@constraint(UC, MDT[i in G_thermal, t in T_MDT(i)], 
            sum((1-u[i,t+1]) for t = t:t+gen_df[gen_df.r_id .== i,:down_time][1]) >= (u[i,t] - u[i,t+1])*min(gen_df[gen_df.r_id .== i,:down_time][1],(T_period[end]-t-1)))

# Demand constraint
@constraint(UC, cDemand[t in T], 
    sum(p[i,t] for i in G_nonthermal) + sum(p2[i,t] for i in G_thermal) == loads_multi[loads_multi.hour .== t, :demand][1])

#= Capacity constraints (non-variable generation)에 u 항목을 추가 -> p 구간을 나눠버려 일단 제외
   p_min*u <= p <= p_max*u 조건인데, p_min*u <= p*u <= p_max*u 조건과 동일한 조건이라고 판단 
@constraint(UC, Cap_thermal_min[i in G_thermal, t in T], 
   p[i,t] >= u[i,t] * gen_df[gen_df.r_id .== i,:existing_cap_mw][1] *
                   gen_df[gen_df.r_id .== i,:min_power][1]) =#

# u와 p의 관계 및 출력제한 설정
@constraint(UC, p1_max[i in G_thermal, t in T], 
   p1[i,t] <= u1[i,t] * gen_df[gen_df.r_id .== i,:existing_cap_mw][1])
@constraint(UC, p2_min[i in G_thermal, t in T], 
   p2[i,t] >= u2[i,t] * gen_df[gen_df.r_id .== i,:existing_cap_mw][1] *
              gen_df[gen_df.r_id .== i,:min_power][1])
@constraint(UC, p2_max[i in G_thermal, t in T], 
   p2[i,t] <= u2[i,t] * gen_df[gen_df.r_id .== i,:existing_cap_mw][1])

# 2. non-variable generation not requiring commitment
@constraint(UC, Cap_nt_nonvar[i in G_nt_nonvar, t in T], 
   p[i,t] <= gen_df[gen_df.r_id .== i,:existing_cap_mw][1])

# Variable generation capacity constraints
@constraint(UC, Cap_var[i in 1:nrow(gen_var_cf)], 
        p[gen_var_cf[i,:r_id], gen_var_cf[i,:hour] ] <= 
                    gen_var_cf[i,:cf] *
                    gen_var_cf[i,:existing_cap_mw])

# Ramp up constraints
@constraint(UC, RampUp[i in G, t in T_red], 
    p[i,t+1] - p[i,t] <= gen_df[gen_df.r_id .== i,:existing_cap_mw][1] * 
                             gen_df[gen_df.r_id .== i,:ramp_up_percentage][1] )

# Ramp down constraints
@constraint(UC, RampDn[i in G, t in T_red], 
    p[i,t] - p[i,t+1] <= gen_df[gen_df.r_id .== i,:existing_cap_mw][1] * 
                             gen_df[gen_df.r_id .== i,:ramp_dn_percentage][1] )

# Solve statement (! indicates runs in place)
optimize!(UC)

# Dataframe of optimal decision variables
solution = value_to_df_puvw(value.(p), value.(u), value.(v), value.(w))
solution = outerjoin(solution[1], solution[2], solution[3],solution[4], on = [:r_id,:hour])
sort!(solution, [:r_id,:hour])
    return (
        solution = solution, 
        cost = objective_value(UC),
    )
end

n=260
T_period = 1800:2000
loads_multi = loads[in.(loads.hour,Ref(T_period)),:]
gen_variable_multi = gen_variable_long[in.(gen_variable_long.hour,Ref(T_period)),:]

gen_df_sens = copy(gen_df)
gen_df_sens[gen_df_sens.resource .== "solar_photovoltaic",
    :existing_cap_mw] .= 3500

gen_df

solution = unit_commitment_multi_time(gen_df, loads_multi, gen_variable_multi, 1e-9);

solution.solution
solution.cost

CSV.write("solution.csv", solution.solution)
marginals = marginals

solution. dual.(RampUp)

sol_gen = innerjoin(solution.solution, 
                    gen_df[!, [:r_id, :resource]], 
                    on = :r_id)
sol_gen = combine(groupby(sol_gen, [:resource, :hour]), 
            :gen => sum)
sol_gen_btm = sol_gen
sol_gen_btm[sol_gen_btm.resource .== "solar_photovoltaic", :resource] .= "_solar_photovoltaic"
sol_gen_btm[sol_gen_btm.resource .== "onshore_wind_turbine", :resource] .= "_onshore_wind_turbine"
sol_gen_btm[sol_gen_btm.resource .== "small_hydroelectric", :resource] .= "_small_hydroelectric"

btm = DataFrame(resource = repeat(["_solar_photovoltaic_btm"]; outer=length(T_period)), 
    hour = T_period,
    gen_sum = gen_variable_multi[gen_variable_multi.gen_full .== "wec_sdge_solar_photovoltaic_1.0",:cf] * 600)
append!(sol_gen_btm, btm)