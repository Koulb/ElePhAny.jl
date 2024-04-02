abstract type AbstractElectrons end

struct Electrons <: AbstractElectrons
    U_list::Array{}
    V_list::Array{}
    ϵkᵤ_list::Array{}
    ϵₚ_list::Array{}#Float64
    ϵₚₘ_list::Array{}
    k_list::Array{}
end

