using Documenter, ElectronPhonon

makedocs(
    sitename="ElectronPhonon.jl",
    authors="Aleksandr Poliukhin",
    clean=true,
    modules=[ElectronPhonon],
    checkdocs=:exports
)
